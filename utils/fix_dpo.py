import json

file_path = "LLaMA-Factory/data/deepseek_math_ul2_dpo.json" 
data = json.load(open(file_path))

for idx, sample in enumerate(data):
    if "<SEP>" not in sample["chosen"]:
        data[idx]["chosen"] = "<SEP>" + sample["chosen"]
    if "<SEP>" not in sample["rejected"]:
        data[idx]["rejected"] = "<SEP>" + sample["rejected"]

with open(file_path, "w") as f:
    json.dump(data, f, indent="\t")

