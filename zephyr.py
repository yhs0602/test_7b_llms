import json

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


class BgColors:
    USER_PROMPT = '\033[44m'  # Blue background
    LLM_OUTPUT = '\033[42m'  # Green background
    RESET = '\033[0m'  # Reset to default


model_name_or_path = "TheBloke/zephyr-7B-beta-AWQ"
device = "cuda"  # the device to load the model onto
custom_tokenizer_config_path = 'zephyr_tokenizer_config.json'  # Adjust this path
# Load the custom tokenizer configuration
with open(custom_tokenizer_config_path, 'r') as f:
    tokenizer_config = json.load(f)

max_tokens = 3000


def main():
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False, **tokenizer_config)
    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant",
         "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    ]

    current_token_count = 0

    while True:
        # Step 1. Get prompt from the user
        print(f"{BgColors.USER_PROMPT}Enter your prompt (type 'END' on a new line to finish):{BgColors.RESET}")
        lines = []
        while True:
            line = input()
            if line == "END":
                break
            lines.append(line)
        prompt = "\n".join(lines)
        if prompt.lower() == "exit":
            break
        # Step 2. Add the prompt to the conversation
        messages.append({"role": "user", "content": prompt})
        # Step 3. Generate
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        token_count = encodeds.size(1)
        # Remove oldest messages if token limit is exceeded
        while token_count > max_tokens and messages:
            messages.pop(0)
            messages.pop(0)
            # Conversation roles must alternate user/assistant/user/assistant/...
            encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
            token_count = encodeds.size(1)

        model_inputs = encodeds.to(device)
        generated_ids = model.generate(
            model_inputs,
            do_sample=True,
            max_new_tokens=512,
        )
        decoded = tokenizer.batch_decode(generated_ids)
        # Step 4. Process the generated output
        chatbot_output = decoded[0]

        # Free memory
        del model_inputs
        torch.cuda.empty_cache()

        # From the last <|assistant|> until the first </s>
        # Find the last <|assistant|>
        position = chatbot_output.rfind("<|assistant|>")
        if position != -1:
            start_position = position + len("[<|assistant|>")
            chatbot_output = chatbot_output[start_position:]
            chatbot_output = chatbot_output.split("</s>")[0]
        else:
            print(f"Failed to parse the output: {chatbot_output}")
            chatbot_output = "OK."
        # Step 5. Print the output
        print(f"{BgColors.LLM_OUTPUT}{chatbot_output}{BgColors.RESET}")
        # Step 5. Add it to the messages (history)
        messages.append({"role": "assistant", "content": chatbot_output})


if __name__ == '__main__':
    main()
