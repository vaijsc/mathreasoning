import os
import numpy as np
import operator
import json
from util import clean_numbers, last_boxed_only, last_boxed_only_string
from math_equivalence import is_equiv
import argparse

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def main(args):
    gt_data = json.load(open(args.gt))

    pred_data = json.load(open(args.pred))
    gt_answers = []

    gt_answers = [remove_boxed(last_boxed_only_string(gt_problem["output"])) for gt_problem in gt_data]

    pred_answers = [remove_boxed(last_boxed_only_string(pred_answer)) for pred_answer in pred_data]

    assert len(gt_answers) == len(pred_answers), f"prediction, groundtruth length mismatch: {len(pred_answers)}, {len(gt_answers)}"
    
    res = []
    for model_output, gt_answer in zip(pred_answers, gt_answers):
        try:
            if is_equiv(model_output, gt_answer):
                res.append(1)
        except:
            res.append(0)
#    print(gt_answers)

    print("Acc", sum(res)/len(gt_answers))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pred", type=str, default="infer_res/ul2_700_real.json")
    parser.add_argument("--gt", type=str, default="LLaMA-Factory/data/math_test.json")
    parser.add_argument("--out_folder", type=str, default="err_analysis/")
    
    args = parser.parse_args()
    main(args)
