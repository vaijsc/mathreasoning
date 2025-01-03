from datasets import Dataset, load_dataset
import json
import argparse
import random
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from functools import partial
import os
import re

SENTENCE_PATTERN = r"\s*(?:\d+|[A-Z]|#+).*?(?:[.?!](?!\d)|[\n]|(?=$))" 
EQUATION_PATTERN = r"((?:\s*\$*\d+\s*)(?:.*?)<<(?:.*?)>>(?:\S*\d+)|\s*#+.*(?=$))"

def ul2_map(samples, indices, num_epochs, denoisers, probs, mask_on_prompt, args):
    results = {
                "instruction": [],
                "input": [],
                "output": []
            }
    
    
    for sample_idx in range(len(indices)):
        if args.random_mask:
            # random mask strategy
            for epoch in range(num_epochs):
                instruction, output = _mask_segments(
                                    prompt=samples["instruction"][sample_idx],
                                    solution=samples["output"][sample_idx],
                                    denoisers=denoisers,
                                    probs=probs,
                                    mask_on_prompt=mask_on_prompt,
                                    args=args
                                )
            
                # <SEP> token is in the output part but still receive bidirectional attention mask -> later prefix length should be added with 1
                results["instruction"].append(instruction)
                results["output"].append(output)
                results["input"].append("")
        elif args.causaleqmasking:
            instructions, outputs = _causaleqmasking(
                                                prompt=samples["instruction"][sample_idx],
                                                solution=samples["output"][sample_idx],
                                                args=args
                                    )            
            
            results["instruction"].extend(instructions)
            results["output"].extend(outputs)
            results["input"].extend(["" for _ in range(len(outputs))])
        elif args.mixedcausalsenteqmasking:
            instructions, outputs = _mixedcausalsenteqmasking(
                                                prompt=samples["instruction"][sample_idx],
                                                solution=samples["output"][sample_idx],
                                                args=args
                                    )            
            
            results["instruction"].extend(instructions)
            results["output"].extend(outputs)
            results["input"].extend(["" for _ in range(len(outputs))])
        elif args.bartmixedreverse:
            instructions, outputs = _bartmixedreverse(
                                                prompt=samples["instruction"][sample_idx],
                                                solution=samples["output"][sample_idx],
                                                args=args
                                    )            
            
            results["instruction"].extend(instructions)
            results["output"].extend(outputs)
            results["input"].extend(["" for _ in range(len(outputs))])
        elif args.bartmixed:
            instructions, outputs = _bartmixed(
                                                prompt=samples["instruction"][sample_idx],
                                                solution=samples["output"][sample_idx],
                                                args=args
                                    )            
            
            results["instruction"].extend(instructions)
            results["output"].extend(outputs)
            results["input"].extend(["" for _ in range(len(outputs))])
        elif args.no_mask:
            instructions, outputs = _no_mask(
                                                prompt=samples["instruction"][sample_idx],
                                                solution=samples["output"][sample_idx],
                                                args=args
                                    )            
            
            results["instruction"].extend(instructions)
            results["output"].extend(outputs)
            results["input"].extend(["" for _ in range(len(outputs))])
        else:
            SENTENCE_PATTERN = r"\s*(?:\d+|[A-Z]).*?(?:[.?!](?!\d)|[\n]|(?=$))" 
            for epoch, denoiser in enumerate(denoisers):
                instruction, output = _planning_mask(
                                    prompt=samples["instruction"][sample_idx],
                                    solution=samples["output"][sample_idx],
                                    denoiser=denoiser,
                                    epoch=epoch,
                                    probs=probs,
                                    args=args
                                )

                # <SEP> token is in the output part but still receive bidirectional attention mask -> later prefix length should be added with 1

                for tmp in range(args.duplicate_plan):
                    results["instruction"].append(instruction)
                    results["output"].append(output)
                    results["input"].append("")
    return results

def union(idx1, idx2, parents:list, heights:list):
    par1 = find(idx1, parents)
    par2 = find(idx2, parents)
    
    if par1 == par2:
        return

    if heights[par1] >= heights[par2]:
        parents[idx2] = par1
        heights[idx2] += 1 
    elif heights[par1] < heights[par2]:
        parents[idx1] = par2
        heights[idx2] += 1

def find(idx1, parents):
    if parents[idx1] == idx1:
        return idx1

    cur = idx1
    while cur != parents[cur]:
        cur = parents[cur]

    parents[idx1] = cur
    return cur

def overlap(obj1:list, obj2:list):
    if obj1[1] < obj2[0] or obj2[1] < obj1[0]:
        return False
    return True

def get_disjoint_masked_positions(masked_objs):
    """not all sentences contain at least 1 equation -> merge overlapping sentences & equations"""
    heights = [0 for _ in range(len(masked_objs))]
    parents = [i for i in range(len(masked_objs))]
    
    adjacent = {i: [] for i in range(len(masked_objs))}
    for i in range(len(masked_objs)):
        for j in range(i+1, len(masked_objs)):
            if overlap(masked_objs[j], masked_objs[i]):
                adjacent[i].append(j)
                adjacent[j].append(i)

    for i in range(len(masked_objs)):
        for j in adjacent[i]:
            union(idx1=i, idx2=j, parents=parents, heights=heights)

    groups = {i: None for i in set(parents)}

    for obj_idx, obj in enumerate(masked_objs):
        par = parents[obj_idx]# find(obj_idx, parents=parents)
        if groups[par] is None:
            groups[par] = obj
        else:
            groups[par] = [min(obj[0], groups[par][0]), max(obj[1], groups[par][1])]
    return sorted(list(groups.values()), key=lambda x: x[0])

def _no_mask(prompt, solution, args):
    """<sentinel_tok_1> is always used for sentence generation"""

    sep_token = "<SEP>" if not args.causalprefix else ""
    if not args.reread:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>"]*args.num_epochs, [sep_token + solution]*args.num_epochs
    else:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>" + f"\n\n{prompt}"]*args.num_epochs, [sep_token + solution]*args.num_epochs
    return prefixes, suffixes 

def _bartmixedreverse(prompt, solution, args):
    all_equations = get_pos_regex(pattern=EQUATION_PATTERN, original_text=solution)[::-1]
    all_sentences = get_pos_regex(pattern=SENTENCE_PATTERN, original_text=solution)[::-1]
    
    print("all sentences and all equations")
    for sent in all_sentences:
        print(solution[sent[0]:sent[1]])
    for eq in all_equations:
        print(solution[eq[0]:eq[1]])
    
    # regular sdenoiser
    sep_token = "<SEP>" if not args.causalprefix else ""

    if not args.reread:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>"]*args.num_epochs, [sep_token + solution]*args.num_epochs
    else:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>" + f"\n\n{prompt}"]*args.num_epochs, [sep_token + solution]*args.num_epochs

    cur_sentence = 1
    for sample_idx in range(1, len(all_equations)):
        # print("sample_idx inital sample_tobe_masked_obj", sample_tobe_masked_obj)
        while cur_sentence < len(all_sentences) and not all_sentences[cur_sentence][1] < all_equations[sample_idx][0]:
            sample_tobe_masked_obj = [all_equations[tmp] for tmp in range(sample_idx, len(all_equations))]
            print("overlap?", overlap(sample_tobe_masked_obj[0], all_sentences[cur_sentence]))
            if overlap(sample_tobe_masked_obj[0], all_sentences[cur_sentence]):
                print("before", sample_tobe_masked_obj)
                sample_tobe_masked_obj[0] = all_sentences[cur_sentence]
                print("after", sample_tobe_masked_obj)
            else:
                sample_tobe_masked_obj = [all_sentences[cur_sentence]] + sample_tobe_masked_obj
            # print("before flip", sample_tobe_masked_obj)
            sample_tobe_masked_obj = sample_tobe_masked_obj[::-1]

            print("after flip", sample_idx, sample_tobe_masked_obj)
            # for obj in sample_tobe_masked_obj:
            #     print(solution[obj[0]: obj[1]])

            cur_sentence += 1
            
            sentinel_num = 0
            prefix = ""
            last_masked_position = 0

            for i in range(len(sample_tobe_masked_obj)):
                sentinel_tok = f"<sentinel_tok_{sentinel_num}>"
                prefix += solution[last_masked_position: sample_tobe_masked_obj[i][0]] + sentinel_tok

                last_masked_position = sample_tobe_masked_obj[i][1]
            
            # add last sentinel token
            prefix += solution[last_masked_position:]
            
            if args.reread:
                prefix += f"\n\n{prompt}"

            # <SEP> token to recognize the bidirectional prompt part, include the string part after the last mask
            suffix = sep_token + solution

            tmp = prompt + "\n\n" + prefix

            prefixes.extend([tmp]*args.causal_dupfactor)
            suffixes.extend([suffix]*args.causal_dupfactor)
    return prefixes, suffixes

def _bartmixed(prompt, solution, args):
    """<sentinel_tok_1> is always used for sentence generation"""

    all_equations = get_pos_regex(pattern=EQUATION_PATTERN, original_text=solution)
    all_sentences = get_pos_regex(pattern=SENTENCE_PATTERN, original_text=solution) 
    
    # regular sdenoiser
    sep_token = "<SEP>" if not args.causalprefix else ""

    if not args.reread:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>"]*args.num_epochs, [sep_token + solution]*args.num_epochs
    else:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>" + f"\n\n{prompt}"]*args.num_epochs, [sep_token + solution]*args.num_epochs

    
    cur_sentence = 0
    for sample_idx in range(0, len(all_equations)-1):
        while cur_sentence < len(all_sentences) and not all_equations[sample_idx][1] <= all_sentences[cur_sentence][0]:
            sample_tobe_masked_obj = [all_equations[tmp] for tmp in range(sample_idx, len(all_equations))]
            if overlap(sample_tobe_masked_obj[0], all_sentences[cur_sentence]):
                sample_tobe_masked_obj[0] = all_sentences[cur_sentence]
            else:
                sample_tobe_masked_obj = [all_sentences[cur_sentence]] + sample_tobe_masked_obj
            cur_sentence += 1            
            
            sentinel_num = 0
            prefix = ""
            last_masked_position = 0

            for i in range(len(sample_tobe_masked_obj)):
                sentinel_tok = f"<sentinel_tok_{sentinel_num}>"
                prefix += solution[last_masked_position: sample_tobe_masked_obj[i][0]] + sentinel_tok

                last_masked_position = sample_tobe_masked_obj[i][1] 
            
            # add last sentinel token
            prefix += solution[last_masked_position:] 
            
            if args.reread:
                prefix += f"\n\n{prompt}"

            # <SEP> token to recognize the bidirectional prompt part, include the string part after the last mask
            suffix = sep_token + solution

            tmp = prompt + "\n\n" + prefix

            prefixes.extend([tmp]*args.causal_dupfactor)
            suffixes.extend([suffix]*args.causal_dupfactor)
        
    return prefixes, suffixes

def _mixedcausalsenteqmasking(prompt, solution, args):
    """<sentinel_tok_1> is always used for sentence generation"""

    all_equations = get_pos_regex(pattern=EQUATION_PATTERN, original_text=solution)
    all_sentences = get_pos_regex(pattern=SENTENCE_PATTERN, original_text=solution) 
    
    # regular sdenoiser
    if not args.reread:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>"]*args.num_epochs, ["<SEP><sentinel_tok_0>" + solution + "<sentinel_tok_1>"]*args.num_epochs
    else:
        prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>" + f"\n\n{prompt}"]*args.num_epochs, ["<SEP><sentinel_tok_0>" + solution + "<sentinel_tok_1>"]*args.num_epochs

    cur_sentence = 0
    for sample_idx in range(0, len(all_equations)-1):
        sample_tobe_masked_obj = [all_equations[tmp] for tmp in range(sample_idx, len(all_equations))]

        while cur_sentence < len(all_sentences) and not sample_tobe_masked_obj[0][1] <= all_sentences[cur_sentence][0]:
            if overlap(sample_tobe_masked_obj[0], all_sentences[cur_sentence]):
                sample_tobe_masked_obj[0] = all_sentences[cur_sentence]
            else:
                sample_tobe_masked_obj = [all_sentences[cur_sentence]] + sample_tobe_masked_obj
            cur_sentence += 1            
            
            sentinel_num = 1

            if args.mixedcausalsenteqfrom0:
                sentinel_num = 0

            prefix = ""
            suffix = ""
            last_masked_position = 0

            for i in range(len(sample_tobe_masked_obj)):
                sentinel_tok = f"<sentinel_tok_{sentinel_num}>"
                prefix += solution[last_masked_position: sample_tobe_masked_obj[i][0]] + sentinel_tok

                suffix += sentinel_tok
                sentinel_num += 1

                last_masked_position = sample_tobe_masked_obj[i][1] 
                suffix += solution[sample_tobe_masked_obj[i][0]: last_masked_position]  
            
            # add last sentinel token
            prefix += solution[last_masked_position:] 
            last_sentinel_tok = f"<sentinel_tok_{sentinel_num}>"

            if args.reread:
                prefix += "\n\n" + prompt

            # <SEP> token to recognize the bidirectional prompt part, include the string part after the last mask
            suffix = "<SEP>" + suffix + last_sentinel_tok
            tmp = prompt + "\n\n" + prefix

            prefixes.extend([tmp]*args.causal_dupfactor)
            suffixes.extend([suffix]*args.causal_dupfactor)
        
    return prefixes, suffixes

def _causaleqmasking(prompt, solution, args):
    all_equations = get_pos_regex(pattern=EQUATION_PATTERN, original_text=solution)
    
    prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>"]*args.num_epochs, ["<SEP><sentinel_tok_0>" + solution + "<sentinel_tok_1>"]*args.num_epochs
    for sample_idx in range(len(all_equations) - 1):
        sample_tobe_masked_obj = [all_equations[tmp] for tmp in range(max(0, len(all_equations) - sample_idx - 2), len(all_equations))]
        
        sentinel_num = 1
        prefix = ""
        suffix = ""
        last_masked_position = 0

        already_masked_positions = set()
        for i in range(len(sample_tobe_masked_obj)):
            if sample_tobe_masked_obj[i][0]-1 not in already_masked_positions:
                 sentinel_tok = f"<sentinel_tok_{sentinel_num}>"
                 prefix += solution[last_masked_position: sample_tobe_masked_obj[i][0]] + sentinel_tok
                 suffix += sentinel_tok
                 already_masked_positions.update(range(sample_tobe_masked_obj[i][0], sample_tobe_masked_obj[i][1]))
                 sentinel_num += 1

            last_masked_position = sample_tobe_masked_obj[i][1] 
            suffix += solution[sample_tobe_masked_obj[i][0]: last_masked_position]  
        
        # add last sentinel token
        prefix += solution[last_masked_position:]

        last_sentinel_tok = f"<sentinel_tok_{sentinel_num}>"

        # <SEP> token to recognize the bidirectional prompt part, include the string part after the last mask
        suffix = "<SEP>" + suffix + last_sentinel_tok
        tmp = prompt + "\n\n" + prefix

        prefixes.extend([tmp]*args.causal_dupfactor)
        suffixes.extend([suffix]*args.causal_dupfactor)
        
    return prefixes, suffixes

def _planning_mask(prompt, solution, denoiser, probs, epoch, args=None):
    #    denoiser = random.choices(denoisers, weights=probs)[0]

    mask_sentence, mode = denoiser

    if not mask_sentence:
        tobe_masked_obj = get_pos_regex(pattern=EQUATION_PATTERN, original_text=solution)
        if mode == 1: # mask all equations
            masked_obj_positions = tobe_masked_obj 

    else:
        # don't mask the first sentence
        tobe_masked_obj = get_pos_regex(pattern=SENTENCE_PATTERN, original_text=solution)[1:]
        if mode == 1: # all equations are masked, some sentences are masked randomly
            masked_obj_positions = get_pos_regex(pattern=EQUATION_PATTERN, original_text=solution)
            masked_sent_positions = random.sample(tobe_masked_obj, k=max(1, int(len(tobe_masked_obj)*0.5)))

            masked_obj_positions.extend(list(masked_sent_positions))
            masked_obj_positions = get_disjoint_masked_positions(masked_obj_positions)

    if mode == 0: # mask some final sentences/equations only
        masked_obj_positions = tobe_masked_obj[len(tobe_masked_obj) - max(2, int(len(tobe_masked_obj)*0.5)): len(tobe_masked_obj)] 
    
    sentinel_num = 50 
    target = ""
    last_masked_position = 0
    for start, end in masked_obj_positions:
        target += solution[last_masked_position: start] + f"<sentinel_tok_{sentinel_num}>"
        last_masked_position = end

        if not args.single_planning_token:
            sentinel_num += 1

    target += solution[last_masked_position:]
    
    return "Generate a plan to solve:\n\n" + prompt, f"<PLAN_{epoch}>" + target

def _mask_segments(prompt, solution, denoisers, probs=None, mask_on_prompt=False, args=None):

    # SENTENCE_PATTERN = r"(?<!\d).*?[\n\?!.]+(?!\d)"
    denoiser = random.choices(denoisers, weights=probs if probs is not None else [1/len(denoisers)]*len(denoisers))[0]

    r = denoiser[0]
    mode = denoiser[1]
    assert mode in [0, 1, 2], f"Unsupported masking mode: {mode}"
    
    tobe_masked_obj = None
    if not mask_on_prompt:
        tobe_masked_seq = solution
    else:
        tobe_masked_seq = prompt + "\n\n" + solution 

    if mode == 0: # s denoiser ~ normal sft
        return f"{prompt}\n\n<sentinel_tok_0>", f"<SEP><sentinel_tok_0>{solution}<sentinel_tok_1>"
    elif mode == 1: # mode == 1: # mask_sentence
        tobe_masked_obj = get_pos_regex(SENTENCE_PATTERN, tobe_masked_seq)
    elif mode == 2: # mask_equation
        tobe_masked_obj = get_pos_regex(EQUATION_PATTERN, tobe_masked_seq)
    if len(tobe_masked_obj) == 0:
        tobe_masked_obj = get_pos_regex(SENTENCE_PATTERN, tobe_masked_seq)

    # Create the masked list
    if r != 1.0:
        num_masked_obj = max(1, int(r*len(tobe_masked_obj)))

        num_potential_masked_objs = len(tobe_masked_obj)
        # always mask the result and the penultimate step equation
        masked_obj_indices = random.sample(
                                            range(   
                                                    0, 
                                                 max(1, num_potential_masked_objs-1) if mode == 1 
                                                 else max(1, num_potential_masked_objs-2)
                                                 ), 
                                            num_masked_obj
                                           )
        if mode == 1:
            masked_obj_indices.append(num_potential_masked_objs-1)
        else:
            masked_obj_indices.extend([num_potential_masked_objs-1, num_potential_masked_objs-2])
    else:
        masked_obj_indices = [i for i in range(len(tobe_masked_obj))]

    masked_obj_indices = list(set(masked_obj_indices))

    if args.sent0sdenoise:
        if mode == 1 and args.sent1sentdenoise:
            sentinel_num = 1
        elif mode == 2 and args.sent1sentdenoise:
            sentinel_num = 100
        else:
            sentinel_num = 1
    else:
        sentinel_num = 0

    prefix = ""
    suffix = ""
    
    last_masked_position = 0 # ensure inclusion of the part until the first match
    
    already_masked_positions = set()
    for i in range(len(tobe_masked_obj)):
        if i in masked_obj_indices:
            if tobe_masked_obj[i][0]-1 not in already_masked_positions:
                 sentinel_tok = f"<sentinel_tok_{sentinel_num}>"
                 prefix += tobe_masked_seq[last_masked_position: tobe_masked_obj[i][0]] + sentinel_tok
                 suffix += sentinel_tok
                 already_masked_positions.update(range(tobe_masked_obj[i][0], tobe_masked_obj[i][1]))
                 sentinel_num += 1

            last_masked_position = tobe_masked_obj[i][1] 
            suffix += tobe_masked_seq[tobe_masked_obj[i][0]: last_masked_position]  
    
    # add last sentinel token
    prefix += tobe_masked_seq[last_masked_position:] 
    last_sentinel_tok = f"<sentinel_tok_{sentinel_num}>"
    # <SEP> token to recognize the bidirectional prompt part, include the string part after the last mask
    suffix = "<SEP>" + suffix + last_sentinel_tok

    if mask_on_prompt:
        return prefix, suffix
    return prompt + "\n\n" + prefix, suffix 

def get_pos_regex(pattern, original_text):
    """Return tuples containing (start, end) of objects (sentence/equation) found by (regex) pattern"""
    result = []
    for match in re.finditer(pattern, original_text):
        start = match.start()
        end = match.end()
        if len(original_text[start: end].strip()) != 0:
            result.append((start, end))
    return result

def convert_pos_and_mask_to_seq_(pos_and_mask, original_seq):
    result = ""
    for tok in pos_and_mask:
        if "<SEP>" in tok or "<sentinel_tok_" in tok:
            result += tok 
        else:
            result += tok + " "
    return result.strip()

def convert_pos_and_mask_to_seq(pos_and_mask, original_seq):
    result = ""
    for tok in pos_and_mask:
        if type(tok) == tuple or type(tok) == list:
            result += original_seq[tok[0]:tok[1]]
        else:
            result += tok
    return result

def main(args, denoisers, probs):
    data = Dataset.from_json(args.dataset)
    
    data = data.map(partial(
                            ul2_map, 
                            num_epochs=args.num_epochs, 
                            denoisers=denoisers,
                            probs=probs,
                            mask_on_prompt=args.mask_on_prompt,
                            args=args
                        ), 
                        with_indices=True,
                        load_from_cache_file=False,
                        batched=True,
                        remove_columns=data.column_names
             )
    
    dataset_name = args.dataset.split("/")[-1].split(".")[0]

    # data.to_json(os.path.join(f"{args.output_path}", f"{dataset_name}_{args.num_epochs}_ul2.json"))
    # Convert the dataset to a list of dictionaries
    samples_list = [sample for sample in data]
    
    if not args.planning:
        outfile_path = os.path.join(f"{args.output_path}", f"{dataset_name}_{args.num_epochs}_ul2")
    else:
        outfile_path = os.path.join(f"{args.output_path}", f"{dataset_name}_ul2")

    if args.mask_on_prompt:
        outfile_path += "_maskonprompt" 
    if args.half_sdenoise:
        outfile_path += "_halfsdenoise"
    if args.sent0sdenoise:
        outfile_path += "_sent0sdenoise"
    if args.sent1sentdenoise:
        outfile_path += "_sent1sentdenoise"
    if args.causaleqmasking:
        outfile_path += f"_{args.causal_dupfactor}_causaleqmasking"
    if args.reread:
        outfile_path += "_reread"

    if args.mixedcausalsenteqmasking:
        outfile_path += f"_{args.causal_dupfactor}_mixedcausalsenteqmasking"
        if args.mixedcausalsenteqfrom0:
            outfile_path += "_from0"

    if args.bartmixed:
        outfile_path += f"_{args.causal_dupfactor}_bartmixed"
        
    if args.bartmixedreverse:
        outfile_path += f"_{args.causal_dupfactor}_bartmixedreverse"

    if args.causalprefix:
        outfile_path += "_causalprefix"

    if args.planning:
        outfile_path += f"_{args.duplicate_plan}_planning"
    if args.extreme_equation_masking:
        outfile_path += "_extreme_equation_masking"
    if args.single_planning_token:
        outfile_path += "_single_planning_token"

    if args.no_mask: 
        outfile_path += "_no_mask"

    if args.random_mask: 
        outfile_path += "_random_mask"

    outfile_path += ".json"

    # Save the list of dictionaries to a JSON file
    with open(outfile_path, 'w') as json_file:
        json.dump(samples_list, json_file, indent="\t", ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="original dataset path")
    parser.add_argument("--num_epochs", "-n", type=int, default=5, help="num ul2 training epochs")
    parser.add_argument("--output_path", type=str, default="LLaMA-Factory/data/")
    parser.add_argument("--mask_on_prompt", action="store_true", help="whether or not to mask on the prompt part")
    parser.add_argument("--half_sdenoise", action="store_true")
    parser.add_argument("--sent0sdenoise", action="store_true")
    parser.add_argument("--sent1sentdenoise", action="store_true")
    parser.add_argument("--causaleqmasking", action="store_true")
    parser.add_argument("--mixedcausalsenteqmasking", action="store_true")
    parser.add_argument("--mixedcausalsenteqfrom0", action='store_true')
    parser.add_argument("--bartmixed", action="store_true")
    parser.add_argument("--bartmixedreverse", action="store_true")
    parser.add_argument("--causal_dupfactor", type=int, default=1, help="how many times to duplicate the causal equation masking strategy?")
    parser.add_argument("--extreme_equation_masking", action="store_true") 
    parser.add_argument("--reread", action="store_true")
    parser.add_argument("--causalprefix", action="store_true")
    parser.add_argument("--no_mask", action="store_true")
    parser.add_argument("--random_mask", action="store_true")

    # planning arguments
    parser.add_argument("--planning", action="store_true")
    parser.add_argument("--single_planning_token", action="store_true")
    parser.add_argument("--duplicate_plan", type=int, default=1, help="how many times to dup planning denoisers")

    args = parser.parse_args()

    seed = 19
    random.seed(seed)
    
    if args.planning:
        # denoisers = ((0, 0), (0, 1), (1, 0), (1, 1))
        denoisers = ((0, 1), (1, 1))
        probs = [1/len(denoisers)]*len(denoisers)
    else:
        # (a, b): (rate, {"mask equation": 2, "mask sentence": 1, "normal sft": 0}) 
        if args.extreme_equation_masking:
            denoisers = ((-1, 0), (0.15, 1), (0.5, 1), (0.5, 2), (0.15, 2), (1.0, 2))
        elif args.causaleqmasking or args.mixedcausalsenteqmasking:
            denoisers = ((1.0, 2))
        else:
            denoisers = ((-1, 0), (0.15, 1), (0.5, 1), (0.5, 2), (0.15, 2)) 

        if args.half_sdenoise:
            probs = [0.5] + [1/(len(denoisers)-1)]*(len(denoisers)-1)
        else:
            probs = [1/len(denoisers)]*len(denoisers)

    main(args, denoisers=denoisers, probs=probs)

