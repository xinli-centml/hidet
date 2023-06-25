from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import torch
import time

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")

model = model.cuda().half()
model = torch.compile(model, mode='max-autotune')
prompt = "A robot may not injure a human being or, through inaction"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids.cuda(), max_new_tokens=55)
text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(text)

start = time.time()
for _ in range(10):
    inputs = tokenizer(prompt, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids.cuda(), max_new_tokens=256)
    text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
end = time.time()
time_per_gen = (end - start) / 10
print('{:.3f}'.format(time_per_gen))
