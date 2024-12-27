import json
import argparse
import re

# def extract_answers(dataset):
#     results = []
#     for sample in dataset:
#         try:
#             str_ans = sample.split("####")[-1].strip()
#             str_ans = str_ans.replace(",", "")
#             answer = int(str_ans)
#         except Exception:
#             answer = -1
# 
#         results.append(answer)
#     return results

def extract_answers(dataset):
    results = []
    answer_pattern = r"####\s*([-]?\d+(?:[.,]\d*)*)"
    for sample in dataset:
        try:
            str_ans = re.findall(answer_pattern, sample)[0].strip()
            str_ans = str_ans.replace(",", "")
                
            answer = int(str_ans)
        except Exception:
            answer = -1

        results.append(answer)
    return results

def main(args):
    gt_data = json.load(open(args.gt))
    gt_data_ans = [sample["output"] for sample in gt_data]

    pred_data = json.load(open(args.pred))

    gt_answers = extract_answers(gt_data_ans)# [0:330]
    pred_answers = extract_answers(pred_data)
    
    assert len(gt_answers) == len(pred_answers), f"prediction, groundtruth length mismatch: {len(pred_answers)}, {len(gt_answers)}"

    res = [1 if gt_answers[i] == pred_answers[i] else 0 for i in range(len(gt_answers))]

    print("Acc", sum(res)/len(gt_answers))

    wrong_answers = []
    idx = 0
    for ans, res in zip(gt_answers, res):
        if res == 0:
            tmp = {}
            tmp["q_id"] = idx
            tmp["instruction"] = gt_data[idx]["instruction"]   
            tmp["wrong"] = pred_data[idx]
            tmp["right"] = gt_data[idx]["output"]
            wrong_answers.append(tmp)
        idx += 1

    with open(f"{args.out_folder}/err-{args.pred.split('/')[-1].split('.')[0]}.json", "w") as f:
        json.dump(wrong_answers, f, indent="\t", ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--pred", type=str, default="infer_res/ul2_700_real.json")
    parser.add_argument("--gt", type=str, default="LLaMA-Factory/data/gsm8k_test.json")
    parser.add_argument("--out_folder", type=str, default="err_analysis/")
    
    args = parser.parse_args()
    main(args)
