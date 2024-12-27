import json
import argparse

def main(args):
    gt_ans = json.load(open(args.gt))
    dpo = json.load(open(args.dpo))

    answer_map = {}
    for question in gt_ans:
        key = question["instruction"]
        if key in answer_map:
            print("overlap", key) 
        else:
            answer_map[key] = question["output"]
    
    for idx, question in enumerate(dpo): 
        if type(question["instruction"]) == list:
            question["instruction"] = question["instruction"][0] 

        if len(question["chosen"]) == 0:
            question["chosen"] = answer_map[question["instruction"].split("\n\n")[0]]

        question["chosen"] = "<SEP>" + question["chosen"]
        question["rejected"] = "<SEP>" + question["rejected"]
        dpo[idx] = question

    with open(f"{args.dpo.split('.')[0]}_mended.json", "w") as f:
        json.dump(dpo, f, indent="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gt", type=str)
    parser.add_argument("--dpo", type=str)

    args = parser.parse_args()

    main(args)
