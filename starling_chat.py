import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer


class BgColors:
    USER_PROMPT = "\033[44m"  # Blue background
    LLM_OUTPUT = "\033[42m"  # Green background
    RESET = "\033[0m"  # Reset to default
    BLUE = "\033[34m"  # Blue text
    GREEN = "\033[32m"  # Green text


model_name_or_path = "TheBloke/Starling-LM-7B-alpha-AWQ"
device = "cuda"  # the device to load the model onto

max_tokens = 3000


def generate_response(tokenizer, model, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # .to(device)
    outputs = model.generate(
        input_ids,
        max_length=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text


def main():
    model = AutoAWQForCausalLM.from_quantized(
        model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=False
    )
    messages = [
        # {"role": "User", "content": "What is your favourite condiment?"},
        # {"role": "Assistant",
        #  "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    ]

    while True:
        # Step 1. Get prompt from the user
        print(f"{BgColors.BLUE}Enter your prompt (type 'END' on a new line to finish):")
        lines = []
        while True:
            line = input()
            if line == "END":
                break
            lines.append(line)
        prompt = "\n".join(lines)
        if prompt.lower() == "exit":
            break
        print(f"{BgColors.RESET}")
        # Step 2. Add the prompt to the conversation
        messages.append({"role": "User", "content": prompt})
        # Step 3. Generate
        encodeds = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        token_count = encodeds.size(1)
        # Remove oldest messages if token limit is exceeded
        while token_count > max_tokens and messages:
            messages.pop(0)
            messages.pop(0)
            # Conversation roles must alternate user/assistant/user/assistant/...
            encodeds = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            token_count = encodeds.size(1)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        model_inputs = encodeds.to(device)
        print(f"{BgColors.GREEN}")
        generated_ids = model.generate(
            model_inputs,
            streamer=streamer,
            do_sample=True,
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        print(f"{BgColors.RESET}")
        decoded = tokenizer.batch_decode(generated_ids)
        # Step 4. Process the generated output
        chatbot_output = decoded[0]
        sayings = chatbot_output.split("<|end_of_turn|>")
        # print(sayings)
        sayings = [saying for saying in sayings if saying.strip()]
        chatbot_final_output = sayings[-1]
        chatbot_final_output = chatbot_final_output.replace(
            " GPT4 Correct Assistant: ", ""
        )
        # Free memory
        del model_inputs
        torch.cuda.empty_cache()

        # Step 5. Print the output
        # print(f"{BgColors.LLM_OUTPUT}{chatbot_final_output}{BgColors.RESET}")
        # Step 5. Add it to the messages (history)
        messages.append({"role": "Assistant", "content": chatbot_final_output})


if __name__ == "__main__":
    main()
