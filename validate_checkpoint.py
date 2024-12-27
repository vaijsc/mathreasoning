import torch

def get_extended_embeddings_state_dict(state_dict, extra_param_name="extra_token_embeddings"):
    # load embedding layer
    result = {}
    for key in state_dict.keys():
        if extra_param_name in key:
            tmp = key.split(".")
            param_name = f"{tmp[-2]}.{tmp[-1]}"

            result[param_name] = state_dict[key]
    return result

if __name__ == "__main__":
    # ckpt = get_extended_embeddings_state_dict(torch.load("saves/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken/checkpoint-580/global_step580/mp_rank_00_model_states.pt"))
    ckpt_ = get_extended_embeddings_state_dict(torch.load("saves/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken/checkpoint-116/global_step116/mp_rank_00_model_states.pt")["module"])
    ckpt_orig = get_extended_embeddings_state_dict(torch.load("saves/deepseek-math-ul2-gsm8k-septoken-maskfull-sentence-equation-lossfulltarget-5ep-bartmixed-singlemasktoken/checkpoint-580/global_step580/mp_rank_00_model_states.pt")["module"])
    keys = list(ckpt_orig.keys())
    print(keys[0])
    print(ckpt_[keys[-2]])
    print(ckpt_orig[keys[-2]])
