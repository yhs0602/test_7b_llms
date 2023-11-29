from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
if __name__ == '__main__':
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                          trust_remote_code=False, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    while True:
        prompt = input("Enter a prompt>")
        if prompt == "exit":
            break
        # <s>[INST] {prompt} [/INST]
        prompt_template=f'''{prompt} 
    
    '''
        print("\n\n*** Generate:")
        tokens = tokenizer(
            prompt_template,
            return_tensors='pt'
        ).input_ids.cuda()
        generation_output = model.generate(
            tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.85,
            top_k=20,
            max_new_tokens=512,
            repetition_penalty=1.2,
        )
        print("Output: ", tokenizer.decode(generation_output[0]))




