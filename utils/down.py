from huggingface_hub import snapshot_download
import argparse

def main(args):
    flag = True

    while flag:
        try:
            print("DOWNLOADING")
            snapshot_download(repo_id=args.model_tag, repo_type=args.repotype, resume_download=True, local_dir=args.model_tag.split("/")[1], local_dir_use_symlinks=False)
        except Exception:
            flag = True
            continue
        finally:
            flag = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default="CohereForAI/c4ai-command-r-v01-4bit")
    parser.add_argument("--repotype", type=str, default="dataset")

    args = parser.parse_args()
    main(args)
