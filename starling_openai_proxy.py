import time
import uuid

import torch
from awq import AutoAWQForCausalLM
from flask import Flask
from flask import request
from transformers import AutoTokenizer

app = Flask(__name__)

model_name_or_path = "TheBloke/Starling-LM-7B-alpha-AWQ"
device = "cuda"  # the device to load the model onto

model = AutoAWQForCausalLM.from_quantized(
    model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)


@app.route("/")
def index():
    return "Hello from Flask!"


# chat completion
@app.route("/v1/chat/completions", methods=["POST", "GET"])
@app.route("/chat/completions", methods=["POST"])
def chat_completion():
    print("got request for chat completion")
    data = request.json
    print("request data", data)
    # data.model : zephyr-beta
    # messages : [{"role": "User", "content": "blah"}]
    # "fallbacks": [{"zephyr-beta": ["gpt-3.5-turbo"]}],
    # "context_window_fallbacks": [{"zephyr-beta": ["gpt-3.5-turbo"]}],
    # "num_retries": 2,
    # "request_timeout": 10
    messages = data["messages"]
    stream = data["stream"]
    encodeds = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    token_count = encodeds.size(1)
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(
        model_inputs,
        do_sample=True,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_token_count = generated_ids.size(1)
    decoded = tokenizer.batch_decode(generated_ids)
    chatbot_output = decoded[0]
    sayings = chatbot_output.split("<|end_of_turn|>")
    sayings = [saying for saying in sayings if saying.strip()]
    chatbot_final_output = sayings[-1]
    chatbot_final_output = chatbot_final_output.replace(" GPT4 Correct Assistant: ", "")
    del model_inputs
    torch.cuda.empty_cache()
    print("Output:", chatbot_final_output)
    return {
        "object": "chat.completion",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {"content": chatbot_final_output, "role": "Assistant"},
            }
        ],
        "id": f"chatcmpl-{uuid.uuid4()}",
        "created": time.time(),
        "model": "TheBloke/Starling-LM-7B-alpha-AWQ",
        "usage": {
            "completion_tokens": new_token_count - token_count,
            "prompt_tokens": token_count,
            "total_tokens": new_token_count,
        },
    }


# text completion
@app.route("/v1/completions", methods=["POST"])
def completion():
    print("got request for completion")
    data = request.json
    print("request data", data)

    return {
        "warning": "This model version is deprecated. Migrate before January 4, 2024 to avoid disruption of service. Learn more https://platform.openai.com/docs/deprecations",
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": time.time(),
        "model": "TheBloke/Starling-LM-7B-alpha-AWQ",
        "choices": [
            {
                "text": "\n\nThe weather in San Francisco varies depending on what time of year and time",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 16, "total_tokens": 23},
    }


app.run(host="0.0.0.0", port=1234)
