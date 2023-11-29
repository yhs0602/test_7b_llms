from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == '__main__':
    device = "cuda"
    print("From pretrained")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", torch_dtype=torch.float16, use_flash_attention_2=True)
    print("Tokeizer From pretrained")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    prompt = "My favourite condiment is"
    print("Model inputs.to")
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    # print("Model.to")
    model.to(device)
    print("Generating")
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
    print("Decoding")
    print(tokenizer.batch_decode(generated_ids)[0])
