import transformers

device = "cuda"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "berkeley-nest/Starling-LM-7B-alpha"
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    "berkeley-nest/Starling-LM-7B-alpha"
)


# model.to(device) OOM


def generate_response(prompt):
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


if __name__ == "__main__":
    # Single-turn conversation
    prompt = "Hello, how are you?"
    single_turn_prompt = (
        f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
    )
    response_text = generate_response(single_turn_prompt)
    print("Response:", response_text)

    ## Multi-turn conversation
    prompt = "Hello"
    follow_up_question = "How are you today?"
    response = ""
    multi_turn_prompt = f"GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant: {response}<|end_of_turn|>GPT4 Correct User: {follow_up_question}<|end_of_turn|>GPT4 Correct Assistant:"
    response_text = generate_response(multi_turn_prompt)
    print("Multi-turn conversation response:", response_text)

    ### Coding conversation
    prompt = "Implement quicksort using C++"
    coding_prompt = f"Code User: {prompt}<|end_of_turn|>Code Assistant:"
    response = generate_response(coding_prompt)
    print("Coding conversation response:", response)
