import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Default model for v1
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

def chat(message, history):
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(input_ids, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only assistant message
    if "assistant" in response:
        response = response.split("assistant")[-1]

    return response

ui = gr.ChatInterface(chat)

ui.launch()
