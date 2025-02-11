
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

def ul2_map(samples, indices, num_epochs, tokenizer, args):
    results = {
                "instruction": [],
                "output": []
            }
    
    
    for sample_idx in range(len(indices)):
        if args.random_mask:
            # random mask strategy
            instruction, output = _ul2_random(
                                prompt=samples["instruction"][sample_idx],
                                solution=samples["output"][sample_idx],
                                tokenizer=tokenizer,
                                args=args
                            )
        
            # <SEP> token is in the output part but still receive bidirectional attention mask -> later prefix length should be added with 1
            results["instruction"].extend(instruction)
            results["output"].extend(output)
        elif args.mixedcausalsenteqmasking:
            instructions, outputs = _mixedcausalsenteqmasking(
                                                prompt=samples["instruction"][sample_idx],
                                                solution=samples["output"][sample_idx],
                                                args=args
                                    )            
            
            results["instruction"].extend(instructions)
            results["output"].extend(outputs)
    return results

def get_pos_regex(pattern, original_text):
    """Return tuples containing (start, end) of objects (sentence/equation) found by (regex) pattern"""
    result = []
    for match in re.finditer(pattern, original_text):
        start = match.start()
        end = match.end()
        if len(original_text[start: end].strip()) != 0:
            result.append((start, end))
    return result

def overlap(obj1:list, obj2:list):
    if obj1[1] < obj2[0] or obj2[1] < obj1[0]:
        return False
    return True

def _ul2_random(prompt, solution, tokenizer, args):
    prefixes, suffixes = [prompt + "\n\n" + "<sentinel_tok_0>"]*args.num_epochs, ["<SEP><sentinel_tok_0>" + solution + "<sentinel_tok_1>"]*args.num_epochs

    tokenized_solution = tokenizer.encode(solution, add_special_tokens=False)
    
    denoisers = [(0.15, 8), (0.5, 32)] #(mask_ratio, mean span length)
    
    for epoch in range(max(1, int(args.num_epochs // 2))):
        for denoiser in denoisers: # can change to samling by probability latter:
            solution_length = len(tokenized_solution)
            masked_length = int(denoiser[0]* solution_length)
            span_lengths = np.random.poisson(denoiser[1], masked_length) # if all spans are of length 1 -> requires masked_length spans
            
            total_masked_length = 0
            masked_spans = []
            masked_positions = set()
            for length in span_lengths:
                if total_masked_length > masked_length:
                    break

                if length >= solution_length:
                    continue

                start = random.randint(0, solution_length - length - 1)
                for i in range(start, min(start + length, solution_length)): 
                    if i == min(start + length, solution_length) - 1:
                        masked_spans.append((start, i))
                        for j in range(start, i):
                            masked_positions.add(j)
                        total_masked_length += i-start
                        break
                    if i in masked_positions:
                        break
            
            prefix, suffix = f"{prompt}", "<SEP>"
            prev_masked_position = 0
            sentinel_num=0
            masked_spans.sort(key=lambda x: x[0])
            for span in masked_spans:
                prefix += tokenizer.decode(tokenized_solution[prev_masked_position: span[0]]) + f"<sentinel_tok_{sentinel_num}>"
                suffix += f"<sentinel_tok_{sentinel_num}>" + tokenizer.decode(tokenized_solution[span[0]:span[1]])
                prev_masked_position = span[1]
                sentinel_num += 1

            prefix += tokenizer.decode(tokenized_solution[prev_masked_position:])
            suffix += f"<sentinel_tok_{sentinel_num}>"

            prefixes.append(prefix)
            suffixes.append(suffix)

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
        while cur_sentence < len(all_sentences) and not all_equations[sample_idx][1] <= all_sentences[cur_sentence][0]:
            sample_tobe_masked_obj = [all_equations[tmp] for tmp in range(sample_idx, len(all_equations))]
            if overlap(sample_tobe_masked_obj[0], all_sentences[cur_sentence]):
                sample_tobe_masked_obj[0] = all_sentences[cur_sentence]
            else:
                sample_tobe_masked_obj = [all_sentences[cur_sentence]] + sample_tobe_masked_obj
            cur_sentence += 1            
            
            sentinel_num = 1

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

def main(args):
    data = Dataset.from_json(args.dataset)
    
    tokenizer = None
    if args.tokenizer != "":
        assert args.random_mask, "only random_mask requires tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    data = data.map(partial(
                            ul2_map, 
                            num_epochs=args.num_epochs, 
                            tokenizer=tokenizer,
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
    
    outfile_path = os.path.join(f"{args.output_path}", f"{dataset_name}_{args.num_epochs}_ul2")

    if args.mixedcausalsenteqmasking:
        outfile_path += f"_{args.causal_dupfactor}_mixedcausalsenteqmasking"

    if args.random_mask:
        outfile_path += f"_{args.model_name}_random_mask"

    outfile_path += ".json"

    # Save the list of dictionaries to a JSON file
    with open(outfile_path, 'w') as json_file:
        json.dump(samples_list, json_file, indent="\t", ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="", help="original dataset path")
    parser.add_argument("--tokenizer", type=str, default="", help="path to tokenizer")
    parser.add_argument("--model_name", type=str, default="", help="model name")
    parser.add_argument("--num_epochs", "-n", type=int, default=5, help="num ul2 training epochs")
    parser.add_argument("--output_path", type=str, default="LLaMA-Factory/data/")
    parser.add_argument("--mixedcausalsenteqmasking", action="store_true")
    parser.add_argument("--causal_dupfactor", "-c", type=int, default=1)
    parser.add_argument("--reread", action="store_true")
    parser.add_argument("--random_mask", action="store_true")

    args = parser.parse_args()

    seed = 19
    random.seed(seed)
    
    main(args)

