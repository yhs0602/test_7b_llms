from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

conversation_history = ''

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"
if __name__ == '__main__':
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Append the new input to the conversation history
        conversation_history += f'Human: {user_input}\n'

        # Construct the prompt with conversational context
        prompt_template = f"[Chatbot]: {conversation_history} [End of conversation history.]"

        # Generate the tokens for the model
        tokens = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

        # Generate a response from the model
        generation_output = model.generate(
            tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
            max_length=1024,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode the generated tokens and print the model's response
        model_response = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        print(f"Bot: {model_response}")

        # Update the conversation history with the model's response
        conversation_history += f'Chatbot: {model_response}\n'
