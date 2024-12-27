from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen2.5-Math-7B")

test_chat = [{"role": "user", "content": "hello how are you?"}]

conversation = tok.apply_chat_template(test_chat, tokenize=False, add_generation_prompt=True)

print(conversation)
