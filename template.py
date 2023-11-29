import json

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
device = "cuda"  # the device to load the model onto
custom_tokenizer_config_path = 'mistral_tokenizer_config.json'  # Adjust this path
# Load the custom tokenizer configuration
with open(custom_tokenizer_config_path, 'r') as f:
    tokenizer_config = json.load(f)

if __name__ == '__main__':
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False, **tokenizer_config)
    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant",
         "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(device)
    model.to(device)
    generated_ids = model.generate(
        model_inputs,
        do_sample=True,
        max_new_tokens=512,
    )
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded[0])
    # generation_output = model.generate(
    #     tokens,
    #     do_sample=True,
    #     temperature=0.85,
    #     top_p=0.85,
    #     top_k=20,
    #     max_new_tokens=512,
    #     repetition_penalty=1.2,
    # )
