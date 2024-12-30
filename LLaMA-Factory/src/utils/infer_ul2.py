import os
import argparse
import torch
import json
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import defaultdict
from transformers.models.llama import LlamaForCausalLM

from peft import LoraConfig, AutoPeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
# Filter warnings by category
from functools import partial
from torch.nn import Module, Parameter, ParameterList
from typing import List
from llamafactory.data import UL2DataCollatorWith4DAttentionMask, prepare_ul2_4d_attention_mask
import inspect
import re

warnings.filterwarnings("ignore")

def single_turn_greedy(model, inputs, tokenizer, max_new_tokens):
    with torch.no_grad(): 
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            num_return_sequences = 1,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=[tokenizer.eos_token_id]
        )
    
    sequences = outputs.sequences 
    logits = outputs.logits

    return sequences, logits

def compute_confidence_score(logits: torch.Tensor, mode):
    if mode == "greedy":
        scores, _ = logits.softmax(dim=-1).topk(dim=-1, k=2)
        return scores[0] - scores[1]

    elif mode == "entropy":
        prob_dist = logits.softmax(dim=-1)
        epsilon = 1e-9
        prob_dist = prob_dist + epsilon

        neg_entropy = -torch.sum(prob_dist * torch.log(prob_dist), dim=-1)
        
        return neg_entropy

def generate(model, inputs, tokenizer, args, sc_mode="none", for_rethinking=False):
    input_length = inputs["input_ids"].shape[1]

    if sc_mode == "none":
        if args.ul2:
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    prefix_lengths=inputs["prefix_lengths"] if "prefix_lengths" in inputs else None,
                    max_new_tokens = args.max_new_tokens,
                    num_beams = args.beam_size if not for_rethinking else args.rethink_beam_size,
                    num_return_sequences = 1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=[tokenizer.eos_token_id]
                )
        else:
            with torch.no_grad():
                generate_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens = args.max_new_tokens,
                    num_beams = args.beam_size if not for_rethinking else args.rethink_beam_size,
                    num_return_sequences = 1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=[tokenizer.eos_token_id]
                )
        generated_seq = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = not for_rethinking, clean_up_tokenization_spaces = False)
    elif sc_mode == "greedy":
        with torch.no_grad():
            generate_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                prefix_lengths=inputs["prefix_lengths"] if "prefix_lengths" in inputs else None,
                max_new_tokens = args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id]
            )
        generated_seq = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = not for_rethinking, clean_up_tokenization_spaces = False)
    elif sc_mode == "cot":
        # only suppport batch_size = 1
        # swictch to outputing dictionary
        k=9
        if args.ul2:
            allowed_keys = ["input_ids", "attention_mask", "prefix_lengths"]
        else:
            allowed_keys = ["input_ids", "attention_mask"]
        input_length = inputs["input_ids"].shape[1]
        _, first_token_logits = single_turn_greedy(
                                                        model, 
                                                        {k: v for k, v in inputs.items() if k in allowed_keys},
                                                        tokenizer, 
                                                        max_new_tokens=2 if args.ul2 and not args.bart_mixed else 1
                                                    ) # to take sentinel tok 0 for granted
        if args.ul2 and not args.bart_mixed:
            _, first_tokens = first_token_logits[1].topk(dim=-1, k=k) # [B, K]
        else:
            _, first_tokens = first_token_logits[0].topk(dim=-1, k=k) # [B, K]

        first_tokens = first_tokens.reshape(-1).unsqueeze(1)

        # generate the rest
        device = inputs["input_ids"].device
        inputs["input_ids"] = torch.cat((inputs["input_ids"].repeat_interleave(repeats=k, dim=0), first_tokens.to(device)), dim=-1)

        inputs["attention_mask"] = torch.cat((inputs["attention_mask"].repeat_interleave(repeats=k, dim=0), torch.ones((inputs["input_ids"].shape[0], 1)).to(device)), dim=-1) 
        
        if "prefix_lengths" in inputs:
            inputs["prefix_lengths"] = inputs["prefix_lengths"].repeat_interleave(repeats=k, dim=-1)
        
        generated_seqs, generated_logits = single_turn_greedy(  
                                                                model,
                                                                {k: v for k, v in inputs.items() if k in allowed_keys},
                                                                tokenizer,
                                                                max_new_tokens=args.max_new_tokens-1
                                                              )

        generated_seqs = generated_seqs[:, input_length:]
        align_seqs = "####"
        align_seq_tokens = tokenizer(align_seqs, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        
        # compute score for the answer part
        scores = torch.zeros((1, generated_seqs.shape[0])).to(device)
        for sample_idx, sample in enumerate(generated_seqs):
            found = False
            for i in range(0, len(generated_logits), len(align_seq_tokens)):
                if tokenizer.decode(sample[i:min(i+len(align_seq_tokens), sample.shape[0])], add_special_tokens=False) == align_seqs:
                    found = True
                    break

            if not found:
                continue

            score_from_idx = i + len(align_seq_tokens)
            
            answer_length = 0
            for answer_tok_position in range(score_from_idx, len(generated_logits)):
                answer_length += 1
                if sample[answer_tok_position] in tokenizer.all_special_ids:
                    answer_length -= 1
                    break 

                scores[0, sample_idx] += compute_confidence_score(generated_logits[answer_tok_position][sample_idx], mode=args.cot_mode)

            scores[0, sample_idx] /= answer_length

        # print out seqs for debugging
        topk_confidence_scores, top_k_confidence_sols = scores.squeeze().topk(dim=-1, k=k)

        generated_seqs = generated_seqs[top_k_confidence_sols, :]

        # handle special tokens
        if args.bart_mixed:
            chosen_solutions = tokenizer.batch_decode(generated_seqs, skip_special_tokens=True)
        else:
            chosen_solutions = tokenizer.batch_decode(generated_seqs, skip_special_tokens=not for_rethinking)

        # answers ensembling
        ensemble_score = defaultdict(float)
        group_by_answer = defaultdict(list) 
        answer_freq = defaultdict(int)
        answers = extract_answers(chosen_solutions)
        
        wrong_chains = []
        correct_chains = []

        for answer_idx, answer in enumerate(answers): 
            if answer != -1: # wrong generation format
                ensemble_score[answer] += topk_confidence_scores[answer_idx].item()
                group_by_answer[answer].append(chosen_solutions[answer_idx]) 
                answer_freq[answer] += 1
        
        if args.cot_mode == "greedy":
            greatest_score = 0 
            greatest_score_ans = ""
            for key, value in ensemble_score.items():
                if value > greatest_score:
                    greatest_score = value 
                    greatest_score_ans = key

        elif args.cot_mode == "entropy":
            greatest_score = 10000
            greatest_score_ans = ""
            for key, value in ensemble_score.items():
                if value/len(group_by_answer[key]) < greatest_score:
                    greatest_score = value/len(group_by_answer[key]) 
                    greatest_score_ans = key

        print("all answers", chosen_solutions) 
        print("most common answer:", greatest_score_ans)
        print("all scores:", topk_confidence_scores)
        print("greatest_freq:", greatest_score)
        # print("all_solutions:", chosen_solutions)
        print("answer_freq", answer_freq)
        # print("group_by_answer", group_by_answer)
        print("ensemble_score", ensemble_score)

        output_dict = {}

        if args.collect_wrong_chain:

            golden_ans = inputs["gt_answers"][0][0]

            for answer, solution in group_by_answer.items():
                if float(answer) != float(golden_ans):
                    wrong_chains.extend(solution)
                else:
                    correct_chains.extend(solution)

            output_dict["wrong_chains"] = wrong_chains
            output_dict["correct_chains"] = correct_chains

        # greatest_score / sum(topk_confidence_scores) >= args.cot_threshold
        try:
            output_dict["tobe_reviewed_chains"] = [group_by_answer[greatest_score_ans][0]]
        except IndexError:
            output_dict["tobe_reviewed_chains"] = ["<sentinel_tok_0>"]
            output_dict["have_to_review"] = False
            return output_dict                                                                                                                                                                                                                                     
        output_dict["greatest_score"] = greatest_score
        output_dict["ensemble_score"] = ensemble_score 
        if answer_freq[greatest_score_ans] / sum(answer_freq.values()) >= args.cot_threshold:
            print("NO REVIEW NEEDED!")
            output_dict["have_to_review"] = False
            return output_dict

        print("HAVE TO REVIEW!")
        print("tobe_reviewed_chains:", [group_by_answer[greatest_score_ans][0]])
        output_dict["have_to_review"] = True
        return output_dict
    else:
        num_return_sequences = 5
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                prefix_lengths=inputs["prefix_lengths"] if "prefix_lengths" in inputs else None,
                max_new_tokens = args.max_new_tokens,
                num_beams = num_return_sequences,
                num_return_sequences = num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id],
                do_sample=True,
                repetition_penalty=1.2
            )
            
            transition_scores = model.compute_transition_scores(
                                                                outputs.sequences, 
                                                                outputs.scores, 
                                                                outputs.beam_indices, 
                                                                normalize_logits=False
                                                            )
        chosen_indices = []
        current_sample_idx = 0
        for idx in range(0, len(transition_scores), num_return_sequences):
            score_batch = transition_scores[idx: idx + num_return_sequences]
            if args.sc == "mean":
                probs = score_batch.exp().mean(dim=-1)
            elif args.sc == "min":
                probs = score_batch.exp().min(dim=-1)[0]
            elif args.sc == "prod":
                probs = score_batch.exp().prod(dim=-1)   
            else:
                raise RuntimeError(f"Unsupported sc type: {args.sc}")

            chosen_indices.extend(current_sample_idx*num_return_sequences + probs.argmax(dim=0, keepdim=True).cpu().numpy())
            current_sample_idx += 1 
        generate_ids = outputs.sequences

        generated_seq = tokenizer.batch_decode(generate_ids[chosen_indices, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    return {"tobe_reviewed_chains": generated_seq, "have_to_review": True}


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path))

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

def prepare_prompts(prompt_batch , tokenizer: AutoTokenizer, args, for_rethinking=False):
    """a collate_fn for eval dataloader"""
    tokenizer.padding_side="left"

    conversations = []
    if args.ul2_rethinking != "none" or args.sc == "cot":
        original_instructions = []

    batch_gt_answers = []

    for i in range(len(prompt_batch)):
        if args.collect_wrong_chain:
            if not for_rethinking: 
                batch_gt_answers.append(extract_answers([prompt_batch[i]["output"]]))
            else: # from rethinking dict
                batch_gt_answers = prompt_batch[i]["gt_answers"]

        if args.ul2:
            if args.ul2_rethinking != "none" or args.sc == "cot":
                original_instructions.append(prompt_batch[i]["instruction"])

            if for_rethinking:
                conversations.append([{"role": "user", "content": prompt_batch[i]["instruction"]}])
            elif args.reread:
                conversations.append([{"role": "user", "content": prompt_batch[i]["instruction"] + "\n\n<sentinel_tok_0>" + f"\n\n{prompt_batch[i]['instruction']}"}])
            else:
                conversations.append([{"role": "user", "content": prompt_batch[i]["instruction"] + "\n\n<sentinel_tok_0>"}])
        else:
            conversations.append([{"role": "user", "content": prompt_batch[i]["instruction"]}])

    tokenizer.padding_side="left"
    # <SEP> token goes after generation prompt token.
    batch = tokenizer.apply_chat_template(
                                        conversations,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )
    
    if args.causal_prefix and not args.ul2:
        sep_token = ""
    else:
        sep_token = "<SEP>"

    for sample_idx in range(len(batch)):
        batch[sample_idx] = batch[sample_idx] + sep_token 

    batch = tokenizer(
                        batch, 
                        max_length=args.max_tokens - args.max_new_tokens, 
                        padding="longest", 
                        return_tensors="pt", 
                        add_special_tokens=False
                    )

    if not args.causal_prefix:
        prefix_lengths = []
        for sample_idx in range(batch["attention_mask"].shape[0]):
             prefix_lengths.append(torch.sum(batch["attention_mask"][sample_idx] != 0).item())
        batch["prefix_lengths"] = torch.LongTensor(prefix_lengths)# torch.LongTensor([batch["input_ids"].shape[1] * batch["input_ids"].shape[0]])

    if args.ul2_rethinking != "none" and not for_rethinking:
        batch["instruction"] = tokenizer(
                                            original_instructions, 
                                            padding="longest", 
                                            max_length=args.max_tokens-args.max_new_tokens, 
                                            add_special_tokens=False, 
                                            return_tensors='pt'
                                        )["input_ids"]
    
    if args.collect_wrong_chain:
        batch["gt_answers"] = torch.Tensor(batch_gt_answers)

    return batch

def fill_mask(original_seq, rethink_seq):
    pattern = r"sentinel_tok_(\d+)>((?:.|\n)*?)(?=<sentinel_tok_(\d+)>)"
    single_sentinel_pattern = r"(<sentinel_tok_{sentinel_idx}>)"

    fillers = re.findall(pattern=pattern, string=rethink_seq)

    for filler in fillers:
        original_seq = re.sub(pattern=single_sentinel_pattern.format(sentinel_idx=filler[0]), repl="{value}".format(value=filler[1]), string=original_seq)
    
    original_seq = re.sub(r"  ", r" ", original_seq)
    original_seq = re.sub(r"[.][.]", ".", original_seq)
    original_seq = re.sub(r"[,][,]", ",", original_seq)
    # clean up excessive sentinel tok just in case 
    original_seq = re.sub(single_sentinel_pattern.format(sentinel_idx="\d+"), "", original_seq)    

    return original_seq

def extract_answers(dataset):
    results = []
    answer_pattern = r"####\s*([-]?\d+(?:[.,]\d*)*)"
    for sample in dataset:
        try:
            str_ans = re.findall(answer_pattern, sample)[0].strip()
            str_ans = str_ans.replace(",", "")
            answer = float(str_ans)
        except Exception:
            answer = -1

        results.append(answer)
    return results

def get_tobe_masked_objects(generated_seq, mode="eq"):
    #TODO: add another mixed masking mode here pls

    if mode == "eq":
        # EQUATION_PATTERN = r"((?:\s*\$*\d+\s*)(?:.*?)<<(?:.*?)>>(?:\S*\d+)|\s*#+.*(?=$))"
#        EQUATION_PATTERN = r"((?:(?:\s*\$*\d+\s*)(?:.*?)<<(?:.*?)>>(?:\S*\d+))|(?:\s*#+.*(?=$)))"
        EQUATION_PATTERN = r"((?:(?:\s*\$*\d+\s*)(?:.*?)<<(?:.*?)>>(?:\S*\d+))|(####\s*(?:[-]?\d+(?:[.,]\d*)*)))"
        matches = re.finditer(pattern=EQUATION_PATTERN, string=generated_seq)
    elif mode == "sent": 
        SENTENCE_PATTERN = r"\s*(?:\d+|[A-Z]|#+).*?(?:[.?!](?!\d)|[\n]|(?=$))" 
        matches = re.finditer(pattern=SENTENCE_PATTERN, string=generated_seq)

    return [match for match in matches]

def rethinking_mask(generated_seq, tobe_masked_obj: list, sent1sentdenoise: bool=False, bart_mixed=False):
    "mask a number of last (equations/sentence) of a generated answer"
    inference_plan = ""

    if sent1sentdenoise:
        sentinel_num = 100
    else:
        sentinel_num = 1

    if bart_mixed:
        sentinel_num = 0

    last_masked_position = 0 

    for match in tobe_masked_obj:
        start = match.start() 
        inference_plan += generated_seq[last_masked_position:start] + f"<sentinel_tok_{sentinel_num}>"
        last_masked_position = match.end()
        if not bart_mixed:
            sentinel_num += 1

    inference_plan += generated_seq[last_masked_position:]

    return inference_plan
    
def rethinking_step(model, instructions, inference_plans, tokenizer, device, args):
    rethinking_prompts_tensors = prepare_prompts(prompt_batch=instructions, args=args, tokenizer=tokenizer, for_rethinking=True)

    for k, v in rethinking_prompts_tensors.items():
        rethinking_prompts_tensors[k] = v.to(device) 

    # model forward to get rethinking result
    if args.sc == "cot":
        if args.bart_mixed:
            generation_output_dict = generate(model, rethinking_prompts_tensors, tokenizer, args, sc_mode="cot", for_rethinking=True)
            return generation_output_dict
        else:
            generation_output_dict = generate(model, rethinking_prompts_tensors, tokenizer, args, sc_mode="none", for_rethinking=True)
    else:
        generation_output_dict = generate(model, rethinking_prompts_tensors, tokenizer, args, sc_mode=args.sc, for_rethinking=True)

    if not args.bart_mixed:
        for idx, rethink_seq in enumerate(generation_output_dict["tobe_reviewed_chains"]):
            generation_output_dict["tobe_reviewed_chains"][idx] = fill_mask(inference_plans[idx], rethink_seq)
    generation_output_dict["have_to_review"] = True
    
    return generation_output_dict

def overlap(obj1:list, obj2:list):
    if obj1.end() < obj2.start() or obj2.end() < obj1.start():
        return False
    return True

def prepare_pairwise_dpo(instruction, chosen, rejected):
    result = []
    if len(chosen) == 0:                                   
        chosen = [""]                                   
    if len(rejected) == 0:                                   
        return []                                   

    for answer_1 in chosen:
        for answer_2 in rejected:
            result.append({"instruction": instruction, "chosen": answer_1, "rejected": answer_2})

    return result

def eval_loop(model, tokenizer, loader, accelerator, args):
    model.eval()

    if accelerator.is_main_process:
        infer_info = []
        
    for batch_data in tqdm(loader, ncols=0, desc="Batch: "):
        # TEXT GENERATION
        generation_output_dict = generate(model, batch_data, tokenizer, args, sc_mode=args.sc)
        
        generated_seqs = generation_output_dict["tobe_reviewed_chains"]
        have_to_review = generation_output_dict["have_to_review"]
        
        if args.ul2_rethinking != "none":
            instructions = []
            for sample in tokenizer.batch_decode(batch_data["instruction"], skip_special_tokens=True):
                instructions.append(sample) 

        # RETHINKING
        final_output_dict = {}

        if args.collect_wrong_chain:
            if len(generation_output_dict["wrong_chains"]) > 0:
                new_instructions = [instruction + "\n\n<sentinel_tok_0>" for instruction in instructions]

                final_output_dict["pairwise_rating_examples"] = prepare_pairwise_dpo(
                                                                        instruction=new_instructions, 
                                                                        chosen=generation_output_dict["correct_chains"], 
                                                                        rejected=generation_output_dict["wrong_chains"]
                                                                    )
        # prepare for later colllection of examples on the way of revision
        pairwise_rating_examples = []

        if args.ul2_rethinking == "eq" and have_to_review:
            tobe_masked_objects_batch = []
            
            ### consider sample by sample batch
            quota = [] 

            continue_reviewing_all = [True]*len(generated_seqs)

            for sample_idx, generated_seq in enumerate(generated_seqs):
                sample_tobe_masked_objects = get_tobe_masked_objects(generated_seq)
                tobe_masked_objects_batch.append(sample_tobe_masked_objects)
                quota.append(len(sample_tobe_masked_objects)) 

            # regenerate with the masked sequence
            for turn in range(max(quota) - 1):
                included_example = []
                turn_prompts = []
                inference_plans = []
                for sample_idx in range(len(tobe_masked_objects_batch)):
                    sample_tobe_masked_objects = get_tobe_masked_objects(generated_seqs[sample_idx])

                    if quota[sample_idx] == 1 or not continue_reviewing_all[sample_idx]:
                        # don't rethink the last step as it is the conclusion
                        continue

                    included_example.append(sample_idx)
                    
                    sample_inference_plan = rethinking_mask(
                                                generated_seqs[sample_idx], 
                                                tobe_masked_obj=sample_tobe_masked_objects[-quota[sample_idx]:], 
                                                sent1sentdenoise=args.sent1sentdenoise,
                                                bart_mixed=args.bart_mixed
                                            )
                    
                    turn_prompt = instructions[sample_idx] + "\n\n" + sample_inference_plan
                    turn_prompts.append({"instruction": turn_prompt})
                    inference_plans.append(sample_inference_plan)
                    quota[sample_idx] -= 1
                
                if len(turn_prompts) > 0:
                    # generate rethink answers
                    rethinking_output_dict = rethinking_step(model, instructions=turn_prompts, tokenizer=tokenizer, inference_plans=inference_plans, device=batch_data["input_ids"].device, args=args) # check to see if the rethink answer is different from the original one
                    original_answers = extract_answers(generated_seqs[i] for i in included_example)
                    rethink_answers = extract_answers(rethinking_output_dict["tobe_reviewed_chains"])
                     
                    print("\n Generated_seqs:\n", [generation_output_dict["tobe_reviewed_chains"][i] for i in included_example])
                    print("\n Turn rethink prompts:\n", turn_prompts)
                    print("\n Rethink_seqs:\n", rethinking_output_dict["tobe_reviewed_chains"])

                    if not rethinking_output_dict["have_to_review"]:
                        print("ENOUGH!! STOP REVIEWING")

                    for idx, (answer1, answer2) in enumerate(zip(original_answers, rethink_answers)):
                        if answer2 != answer1 and answer2 != -1:
                            generated_seqs[included_example[idx]] = rethinking_output_dict["tobe_reviewed_chains"][idx]        

                    continue_reviewing_all[sample_idx] = rethinking_output_dict["have_to_review"]

        elif args.ul2_rethinking == "mixed" and have_to_review:
            sample_greatest_score = [generation_output_dict["greatest_score"]]*len(generated_seqs)

            for sample_idx in range(len(generated_seqs)):
                sample_tobe_masked_obj = get_tobe_masked_objects(generated_seqs[sample_idx], mode="eq")
                all_sentences = get_tobe_masked_objects(generated_seqs[sample_idx], mode="sent")
                
                cur_sentence = 0
                cur_equation = 0
                try:
                    while not sample_tobe_masked_obj[0].end() <= all_sentences[0].start():
                        if overlap(sample_tobe_masked_obj[0], all_sentences[0]):
                            sample_tobe_masked_obj[0] = all_sentences[0]
                        else:
                            sample_tobe_masked_obj = [all_sentences[0]] + sample_tobe_masked_obj

                        sample_inference_plan = rethinking_mask(
                                                    generated_seqs[sample_idx], 
                                                    tobe_masked_obj=sample_tobe_masked_obj, 
                                                    sent1sentdenoise=False,
                                                    bart_mixed=args.bart_mixed
                                                )

                        while "\n\n" in sample_inference_plan:
                            sample_inference_plan = sample_inference_plan.replace("\n\n", "\n")
                        
                        # PREPARE RETHINKING PROMPT
                        if args.reread:
                            turn_prompt = instructions[sample_idx] + "\n\n" + sample_inference_plan + f"\n\n{instructions[0]}"
                        else:
                            turn_prompt = instructions[sample_idx] + "\n\n" + sample_inference_plan
                        
                        if args.collect_wrong_chain:
                            reviewing_instructions = [{"instruction": turn_prompt, "gt_answers": batch_data["gt_answers"]}]
                        else:
                            reviewing_instructions = [{"instruction": turn_prompt}]

                        rethinking_output_dict = rethinking_step( 
                                                        model,
                                                        instructions=reviewing_instructions,
                                                        tokenizer=tokenizer,
                                                        inference_plans=[sample_inference_plan],
                                                        device=batch_data["input_ids"].device,
                                                        args=args
                                                       )
                        if args.collect_wrong_chain:
                            pairwise_rating_examples.extend(
                                    prepare_pairwise_dpo(
                                                            instruction=turn_prompt, 
                                                            chosen=rethinking_output_dict["correct_chains"],
                                                            rejected=rethinking_output_dict["wrong_chains"]
                                                        )
                                    )
                                                    

                        # check to see if the rethink answer is different from the original one
                        original_answer = extract_answers([generated_seqs[sample_idx]])[0]
                        rethink_answer = extract_answers(rethinking_output_dict["tobe_reviewed_chains"])[0]
                        
                        if rethink_answer != -1 and rethinking_output_dict["greatest_score"] > sample_greatest_score[sample_idx]*(1/args.cot_threshold):
                            generated_seqs[sample_idx] = rethinking_output_dict["tobe_reviewed_chains"][0]

                            sample_greatest_score[sample_idx] = rethinking_output_dict["greatest_score"]
                            print("ORIGINAL_ANSWER:", original_answer) 
                            print("TOBE_REVIEWED_SEQ:", generated_seqs[sample_idx])
                            print("RETHINK_PROMPT", turn_prompt)
                            print("RETHINK_ANSWER:", rethink_answer)
                            print("RETHINK_SEQ:", generated_seqs[sample_idx])
                            cur_sentence = 0
                            cur_equation = 0

                        cur_sentence += 1
                        cur_equation += 1

                        if not rethinking_output_dict["have_to_review"]:
                            print("ENOUGH!! STOP REVIEWING")
                            print("RETHINK_SEQ:", rethinking_output_dict["tobe_reviewed_chains"][0])
                            break
                        
                        sample_tobe_masked_obj = get_tobe_masked_objects(generated_seqs[sample_idx], mode="eq")
                        all_sentences = get_tobe_masked_objects(generated_seqs[sample_idx], mode="sent")

                        if cur_sentence == len(all_sentences) - 1:
                            break

                        sample_tobe_masked_obj = sample_tobe_masked_obj[cur_equation:]
                        all_sentences = all_sentences[cur_sentence:]
                except IndexError:
                    continue

        if args.collect_wrong_chain:
            if "pairwise_rating_examples" in final_output_dict:
                final_output_dict["pairwise_rating_examples"].extend(pairwise_rating_examples)
            else:
                final_output_dict["pairwise_rating_examples"] = pairwise_rating_examples
        else:
            final_output_dict["generated_seqs"] = generated_seqs
        
        all_generated_seqs = accelerator.gather_for_metrics(final_output_dict, use_gather_object=True)
        if accelerator.is_main_process:
            if args.collect_wrong_chain and "pairwise_rating_examples" in final_output_dict:
                infer_info.extend(final_output_dict["pairwise_rating_examples"])
            else:
                infer_info.append(final_output_dict)

    if accelerator.is_main_process:
        if args.collect_wrong_chain:
            return infer_info
        else:
            results = []
            for idx, infer_output_dct in enumerate(infer_info):
                results.extend(infer_output_dct["generated_seqs"])
            return results
    return None


class CustomEmbedding(Module):
    """Custom embedding layers for training with reserved tokens only"""
    def __init__(self, original_embeddings, token_ids: List):
        super(CustomEmbedding, self).__init__()
        self.original_embeddings = original_embeddings
        self.token_ids = token_ids
        self.extra_token_embeddings = ParameterList([Parameter(data=original_embeddings.weight[token_id].clone(), requires_grad=False) for token_id in self.token_ids])

        self.id_map = {}
        for index, token_id in enumerate(self.token_ids):
            self.id_map[token_id] = index

        # freeze original embedding tokens
        self.original_embeddings.weight.requires_grad = False

    def forward(self, input_ids):
        embeddings = self.original_embeddings(input_ids).detach()
        for row_id in range(input_ids.shape[0]):
            for pos, token_id in enumerate(input_ids[row_id]):
                token_id_cpu = token_id.cpu().item()

                if token_id_cpu in self.token_ids:
                    embeddings[row_id, pos, :] = self.extra_token_embeddings[self.id_map[token_id_cpu]]
        return embeddings


def get_extended_embeddings_state_dict(state_dict, extra_param_name="extra_token_embeddings"):
    # load embedding layer
    result = {}
    for key in state_dict.keys():
        if extra_param_name in key:
            tmp = key.split(".")
            param_name = f"{tmp[-2]}.{tmp[-1]}"

            result[param_name] = state_dict[key]
    return result


def extend_vocab(model, tokenizer):
    """
        Replace the model Embedding with the custom embedding to learn newly added tokens only
    """
    num_new_token = 100 # default value

    original_embedding = model.get_input_embeddings() # to get the embedding layer

    original_embedding_vocab_size = original_embedding.weight.shape[0]
    original_vocab_size = len(tokenizer)

    num_reserved_tokens = original_embedding_vocab_size - original_vocab_size

    # add new token 
    num_new_token = max(num_new_token, num_reserved_tokens)

    tokenizer.add_special_tokens({"additional_special_tokens":[f"<sentinel_tok_{i}>" for i in range(num_new_token-21)]+ ["<SEP>"] + [f"<PLAN_{i}>" for i in range(20)]})
    # tokenizer.add_special_tokens({"additional_special_tokens":[f"<sentinel_tok_{i}>" for i in range(num_new_token-1)]+ ["<SEP>"]})
    # print(tokenizer.encode("<sentinel_tok_10>"))
    new_embedding = CustomEmbedding(model.resize_token_embeddings(len(tokenizer)), [original_vocab_size+i for i in range(num_new_token)])

    model.set_input_embeddings(new_embedding)

    return model, tokenizer


def patch_prepare_inputs_for_generation(
        # self,  # setattr for an object doesn't require self as a kwarg
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        prefix_lengths=None,
        **kwargs,
    ):
        # QHP: kwargs should contain "prefix_lengths" to build the attention mask for ul2 generation
        #print("Within patched function")
        #print("kwargs_keys", kwargs.keys())
        #print(input_ids.shape)
        #print("kwargs", prefix_lengths)
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        if prefix_lengths is not None:
            # print("prefix lengths neeee", input_ids.dtype) 
            # print("before convert attn_mask", attention_mask.shape)
            attention_mask = prepare_ul2_4d_attention_mask(attention_mask, prefix_lengths, torch.float16)[:, :, -input_ids.shape[1]:, :]
            # print("prefix_lengths", prefix_lengths)
            # print(input_ids.shape)
            # print("4d_mask", attention_mask.shape)

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_path)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.eos_token = tokenizer.pad_token
    print(len(tokenizer))

    accelerator = Accelerator()
    print("START LOADING MODEL")
    lora_path = args.lora_path

    model = AutoModelForCausalLM.from_pretrained(
                                                    lora_path if len(lora_path) else args.base_path,
                                                    device_map="auto", 
                                                    # attn_implementation="flash_attention_2", 
                                                    torch_dtype=torch.float16
                                                )

    if args.ul2:
        state_dict_path = f"{lora_path}/global_step{lora_path.split('-')[-1]}/mp_rank_00_model_states.pt"
        state_dict = torch.load(state_dict_path)["module"]

        model, tokenizer = extend_vocab(model, tokenizer)
        
        # load extra parameters for embedding layer
        embedding = model.get_input_embeddings()
        embedding.load_state_dict(get_extended_embeddings_state_dict(state_dict, extra_param_name=args.extra_param_name), strict=False)

        if not args.causal_prefix:
            setattr(model, 'prepare_inputs_for_generation', patch_prepare_inputs_for_generation)
    
    eval_set = CustomDataset(args.dataset_path)
    eval_loader = DataLoader(   
                                eval_set, 
                                batch_size=args.batch_size, 
                                collate_fn=partial(
                                                        prepare_prompts, 
                                                        tokenizer=tokenizer, 
                                                        args=args
                                                ),
                                shuffle=False,
                                num_workers=32
                            )

    accelerator.wait_for_everyone()
    print("Loading finished")
      
    model, loader = accelerator.prepare(model, eval_loader)

    results = eval_loop(
                            model=model, 
                            tokenizer=tokenizer,
                            loader=loader, 
                            accelerator=accelerator,
                            args=args
                        )
    
    if accelerator.is_main_process:
        print("LLM name:", args.lora_path)
        with open(f"{args.out_file}", "w") as f:
            json.dump(results, f, indent='\t', ensure_ascii=False)
            print(f"{args.out_file} saved!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default="deepseek-math-7b-base/", type=str, help="path to base model")
    parser.add_argument('--lora_path', default="", type=str, help="path to lora checkpoint empty if full sft")

    parser.add_argument('--dataset_path', default="/home/ubuntu/math_infer/LLaMA-Factory/data/gsm8k_test.json", type = str, help="path to the alpaca dev dataset")

    parser.add_argument('--out_file', default="", type=str, help="out file path")
    
    parser.add_argument("--extra_param_name", type=str, default="extra_token_embeddings")
    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)

    parser.add_argument("--ul2", action="store_true")
    parser.add_argument("--causal_prefix", action="store_true")
    parser.add_argument("--ul2_rethinking", choices=['none', 'mixed', 'eq'], default='none', help="masking strategy for rethinking")
    parser.add_argument("--rethink_beam_size", type=int, default=5)
    parser.add_argument("--reread", action="store_true", help="reread input question")
    parser.add_argument("--bart_mixed", action="store_true", help="is this the bart_mixed model")
    parser.add_argument("--cot_threshold", type=float, default=0.4, help="threshold for not reviewing")
    
    parser.add_argument("--sent1sentdenoise", action="store_true")
    parser.add_argument("--sc", choices=['min', 'mean', 'prod', 'ul2', 'cot', 'none'], default="none", help="whether or not to use soft self-consistency sampling")
    parser.add_argument("--cot_mode", choices=["greedy", "entropy"], default="greedy", help="greedy = top[0] - top[1]")
    parser.add_argument("--collect_wrong_chain", action="store_true")

    args = parser.parse_args()

    print(args)
    if args.ul2_rethinking == "mixed":
        assert args.batch_size == 1, f"Unsupported batch_size {args.batch_size} > 1"

    main(args)

"""accelerate launch --config_file LLaMA-Factory/examples/accelerate/single.yaml utils/infer_large_models.py --llm_path saves/granite_ddl_init44/checkpoint-292/ --dataset_path LLaMA-Factory/data/bird_dev_new_matching_.json --out_file granite_292_infer.json --db_root_path LLaMA-Factory/data/sft_data_collections/bird/dev/dev_databases/ --sc "none"
"""
