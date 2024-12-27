import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp_file", type=str)
    parser.add_argument("--out_file", type=str)

    args = parser.parse_args()

    results = []
    with open(args.inp_file, "r") as f:
        for line in f:
            tmp = json.loads(line)
            results.append({
                                "instruction": tmp["question"],
                                "input": "",
                                "output": tmp["answer"]
                            })
    
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent="\t", ensure_ascii=False)
