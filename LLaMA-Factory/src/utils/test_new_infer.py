import argparse
import os
import torch
import json
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, AutoPeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
from torch.utils.data import Dataset
from accelerate import Accelerator
from accelerate.utils import gather_object
from sql_metadata import Parser
# Filter warnings by category
from functools import partial
from torch.nn import Module, Parameter, ParameterList
from typing import List
from llamafactory.data import UL2DataCollatorWith4DAttentionMask, prepare_ul2_4d_attention_mask
import inspect
import re

warnings.filterwarnings("ignore")

def generate(model, inputs, tokenizer, args):
    input_length = inputs["input_ids"].shape[1]

    if args.sc == "none":
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens = args.max_new_tokens,
                num_beams = args.beam_size,
                num_return_sequences = args.beam_size,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id]
            )
        generated_seq = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    else:
        num_return_sequences = 9
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens = args.max_new_tokens,
                num_beams = num_return_sequences,
                num_return_sequences = num_return_sequences,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id],
                do_sample=True
            )
            
            transition_scores = model.compute_transition_scores(
                                                                outputs.sequences, 
                                                                outputs.scores, 
                                                                outputs.beam_indices, 
                                                                normalize_logits=False
                                                            )
        chosen_indices = [] 
        for idx in range(0, len(transition_scores), num_return_sequences):
            score_batch = transition_scores[idx: idx + num_return_sequences]
            if args.sc == "mean":
                probs = score_batch.exp().mean(dim=1)
            elif args.sc == "min":
                probs = score_batch.exp().min(dim=1)[0]
            elif args.sc == "prod":
                probs = score_batch.exp().prod(dim=1)   
            else:
                raise RuntimeError(f"Unsupported sc type: {args.sc}")

            chosen_indices.extend(probs.argmax(dim=0, keepdim=True).cpu().numpy())
        
        generate_ids = outputs.sequences

        generated_seq = tokenizer.batch_decode(generate_ids[chosen_indices, input_length:], skip_special_tokens = True, clean_up_tokenization_spaces = False)
    """ print("Gen tot': ", generated_seq) """
    return generated_seq


class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path))

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

def prepare_prompts(prompt_batch , tokenizer: AutoTokenizer, max_tokens, max_new_tokens, args):
    tokenizer.padding_side="left"

    conversations = []

    for i in range(len(prompt_batch)):
        conversations.append([{"role": "user", "content": prompt_batch[i]["instruction"]}])

    tokenizer.padding_side="left"
    
    pattern = r"<sentinel_tok_(\d+)>"
    for sample_idx in range(len(conversations)):
        review_mode = re.findall(pattern, conversations[sample_idx][0]["content"])
        if not review_mode:
            conversations[sample_idx][0]["content"] = conversations[sample_idx][0]["content"] + f"\n\n<sentinel_tok_0>" 
    # <SEP> token goes after generation prompt token.
    batch = tokenizer.apply_chat_template(
                                        conversations,
                                        tokenize=False,
                                        add_generation_prompt=True
                                    )

    for sample_idx in range(len(batch)):
        batch[sample_idx] = batch[sample_idx] + "<SEP>"
    print("input_prompt", batch)
    batch = tokenizer(
                        batch, 
                        max_length=max_tokens - max_new_tokens, 
                        padding="longest", 
                        return_tensors="pt", 
                        add_special_tokens=False
                    )
    
    if args.ul2:
        prefix_lengths = []
        for sample_idx in range(batch["attention_mask"].shape[0]):
             prefix_lengths.append(torch.sum(batch["attention_mask"][sample_idx] != 0).item())
        batch["prefix_lengths"] = torch.LongTensor(prefix_lengths)# torch.LongTensor([batch["input_ids"].shape[1] * batch["input_ids"].shape[0]])
    return batch


def eval_loop(model, tokenizer, loader, accelerator, args):
    model.eval()

    if accelerator.is_main_process:
        infer_info = []
        
    for batch_data in tqdm(loader, ncols=0, desc="Batch: "):
        # print(batch_data["attention_mask"].shape, batch_data["input_ids"].shape)

        generated_seqs = generate(model, batch_data, tokenizer, args)
        all_generated_seqs = accelerator.gather_for_metrics(generated_seqs, use_gather_object=True)
        if accelerator.is_main_process:
            infer_info.extend(all_generated_seqs)

    if accelerator.is_main_process:
        results = []
        for idx, seq in enumerate(infer_info):
            results.append(seq)
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


def extend_vocab(model, tokenizer, load_extra_param):
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
    
    # print(tokenizer.encode("<sentinel_tok_10>"))
    if load_extra_param:
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
            attention_mask = prepare_ul2_4d_attention_mask(attention_mask, prefix_lengths, torch.float16)[:, :, -input_ids.shape[1]:, :]

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
                                                    lora_path, 
                                                    device_map="auto", 
                                                    # attn_implementation="flash_attention_2", 
                                                    torch_dtype=torch.float16
                                                )

    if args.ul2:
        if args.load_extra_param: 
            state_dict_path = f"{lora_path}/global_step{lora_path.split('-')[-1]}/mp_rank_00_model_states.pt"
            state_dict = torch.load(state_dict_path)["module"]

        model, tokenizer = extend_vocab(model, tokenizer, load_extra_param=args.load_extra_param)
        
        # load extra parameters for embedding layer
        if args.load_extra_param:
            embedding = model.get_input_embeddings()
            embedding.load_state_dict(get_extended_embeddings_state_dict(state_dict, extra_param_name=args.extra_param_name), strict=False)

        setattr(model, 'prepare_inputs_for_generation', patch_prepare_inputs_for_generation)
    
    accelerator.wait_for_everyone()
    print("Loading finished")
      
    model = accelerator.prepare(model)
    eval_set = CustomDataset(args.dataset_path)

    while True:
        inp = input()
        test_sample_idx = -1
        try:
            test_sample_idx = int(inp) 
            test_sample = [eval_set[test_sample_idx]]
        except ValueError:
            test_sample = [{"instruction": open(args.text_file, "r").read().strip()}]  

        test_sample = prepare_prompts(
                                        test_sample, 
                                        tokenizer=tokenizer, 
                                        max_tokens=args.max_tokens, 
                                        max_new_tokens=args.max_new_tokens, 
                                        args=args
                                    )
        
        input_length = test_sample["input_ids"].shape[1]
        
        for key, value in test_sample.items():
            test_sample[key] = value.to(model.device)
            
        with torch.no_grad():
            generate_ids = model.generate(
                **test_sample,
                max_new_tokens = args.max_new_tokens,
                num_beams = args.beam_size,
                num_return_sequences = 1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=[tokenizer.eos_token_id],
                repetition_penalty=1.5
            )
        generated_seq = tokenizer.batch_decode(generate_ids[:, input_length:], skip_special_tokens = False, clean_up_tokenization_spaces = False)

        print(generated_seq)
        if test_sample_idx != -1:
            print(eval_set[test_sample_idx]["instruction"])
            print("Golden_ans:\n", eval_set[test_sample_idx]["output"])

        del test_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default="/home/hieupq1/hieupq1/math/deepseek-math-7b-base/", type=str, help="path to base model")
    parser.add_argument('--lora_path', default="/home/hieupq1/hieupq1/math/saves/deepseek-math-ul2-debug/checkpoint-232", type=str, help="path to lora checkpoint empty if full sft")
    # parser.add_argument('--state_dict_path', default="", type=str, help="path to satedict for loading extra parameters")

    parser.add_argument('--dataset_path', default="/home/hieupq1/hieupq1/math/LLaMA-Factory/data/gsm8k_test.json", type = str, help="path to the alpaca dev dataset")

    parser.add_argument('--out_file', default="", type=str, help="out file path")
    
    parser.add_argument("--extra_param_name", type=str, default="extra_token_embeddings")
    parser.add_argument("--load_extra_param", action="store_true", help="Did your training include finetuning extra paramenters?") 
    parser.add_argument('--max_tokens', type = int, default = 4096)
    parser.add_argument('--max_new_tokens', type = int, default = 512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--ul2", action="store_true")
    parser.add_argument("--sc", choices=['min', 'mean', 'prod', 'ul2', 'none'], default="none", help="whether or not to use soft self-consistency sampling")
    parser.add_argument("--text_file", type=str, default="./test.txt", help="path to test text file")
    args = parser.parse_args()

    print(args)
    main(args)

