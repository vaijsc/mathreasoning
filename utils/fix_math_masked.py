import json

fp="LLaMA-Factory/data/MATH_masked_train.json"
samples = json.load(open(fp))
for sample in samples:
    sample["instruction"] = sample["instruction"].replace("\n\nsentinel_tok_0", "\n\n<sentinel_tok_0>")
    print(sample["instruction"])

json.dump(samples, open(fp, "w"))
