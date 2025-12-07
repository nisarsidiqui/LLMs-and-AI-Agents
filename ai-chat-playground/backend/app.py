from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from typing import List, Dict, Optional
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Chat history storage (in production, use a database)
chat_history: Dict[str, List[Dict]] = {}

# Model configurations
MODEL_CONFIGS = {
    "llama-3-8b": {
        "type": "huggingface",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "requires_auth": True
    },
    "llama-3-70b": {
        "type": "huggingface",
        "model_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "requires_auth": True
    },
    "mistral-7b": {
        "type": "huggingface",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "requires_auth": False
    },
    "llama-2-7b": {
        "type": "huggingface",
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "requires_auth": True
    },
    "groq-llama-3-8b": {
        "type": "groq",
        "model_id": "llama-3-8b-8192"
    },
    "groq-llama-3-70b": {
        "type": "groq",
        "model_id": "llama-3-70b-8192"
    },
    "groq-mixtral": {
        "type": "groq",
        "model_id": "mixtral-8x7b-32768"
    }
}

# Global model instances (lazy loading)
model_instances = {}


def get_huggingface_model(model_id: str, requires_auth: bool = False):
    """Load and return a Hugging Face model"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        cache_key = f"hf_{model_id}"
        if cache_key in model_instances:
            return model_instances[cache_key]
        
        logger.info(f"Loading Hugging Face model: {model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HUGGINGFACE_TOKEN") if requires_auth else None
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=os.getenv("HUGGINGFACE_TOKEN") if requires_auth else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        if not torch.cuda.is_available():
            model = model.to("cpu")
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_instances[cache_key] = {
            "model": model,
            "tokenizer": tokenizer
        }
        
        logger.info(f"Model {model_id} loaded successfully")
        return model_instances[cache_key]
        
    except Exception as e:
        logger.error(f"Error loading Hugging Face model: {str(e)}")
        raise


def get_groq_client():
    """Get Groq API client"""
    try:
        from groq import Groq
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        if "groq_client" not in model_instances:
            model_instances["groq_client"] = Groq(api_key=api_key)
        
        return model_instances["groq_client"]
        
    except Exception as e:
        logger.error(f"Error initializing Groq client: {str(e)}")
        raise


def format_chat_prompt(messages: List[Dict], model_id: str) -> str:
    """Format chat messages into a prompt based on model type"""
    if "llama-3" in model_id.lower():
        # Llama 3 format
        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    elif "llama-2" in model_id.lower():
        # Llama 2 format
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"[INST] {content} [/INST] "
            elif role == "assistant":
                prompt += f"{content} "
        return prompt
    elif "mistral" in model_id.lower():
        # Mistral format
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content} </s>"
        return prompt
    else:
        # Default format
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{role.capitalize()}: {content}\n"
        prompt += "Assistant: "
        return prompt


def generate_huggingface_response(messages: List[Dict], model_id: str, config: Dict) -> str:
    """Generate response using Hugging Face model"""
    try:
        model_data = get_huggingface_model(model_id, config.get("requires_auth", False))
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        
        # Format prompt
        prompt = format_chat_prompt(messages, model_id)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to device
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error generating Hugging Face response: {str(e)}")
        raise


def generate_groq_response(messages: List[Dict], model_id: str) -> str:
    """Generate response using Groq API"""
    try:
        client = get_groq_client()
        
        # Convert messages to Groq format
        groq_messages = []
        for msg in messages:
            groq_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        # Make API call
        response = client.chat.completions.create(
            model=model_id,
            messages=groq_messages,
            temperature=0.7,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating Groq response: {str(e)}")
        raise


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


@app.route("/models", methods=["GET"])
def get_models():
    """Get list of available models"""
    return jsonify({
        "models": list(MODEL_CONFIGS.keys()),
        "configs": MODEL_CONFIGS
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        message = data.get("message", "")
        model_name = data.get("model", "mistral-7b")
        session_id = data.get("session_id", "default")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        if model_name not in MODEL_CONFIGS:
            return jsonify({"error": f"Model {model_name} not found"}), 400
        
        # Initialize chat history for session
        if session_id not in chat_history:
            chat_history[session_id] = []
        
        # Add user message to history
        chat_history[session_id].append({
            "role": "user",
            "content": message
        })
        
        # Get model config
        config = MODEL_CONFIGS[model_name]
        
        # Generate response based on model type
        if config["type"] == "huggingface":
            response_text = generate_huggingface_response(
                chat_history[session_id],
                config["model_id"],
                config
            )
        elif config["type"] == "groq":
            response_text = generate_groq_response(
                chat_history[session_id],
                config["model_id"]
            )
        else:
            return jsonify({"error": f"Unsupported model type: {config['type']}"}), 400
        
        # Add assistant response to history
        chat_history[session_id].append({
            "role": "assistant",
            "content": response_text
        })
        
        return jsonify({
            "response": response_text,
            "session_id": session_id,
            "model": model_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat/history", methods=["GET"])
def get_chat_history():
    """Get chat history for a session"""
    session_id = request.args.get("session_id", "default")
    return jsonify({
        "history": chat_history.get(session_id, []),
        "session_id": session_id
    }), 200


@app.route("/chat/clear", methods=["POST"])
def clear_chat_history():
    """Clear chat history for a session"""
    data = request.json
    session_id = data.get("session_id", "default")
    chat_history[session_id] = []
    return jsonify({"message": "Chat history cleared", "session_id": session_id}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)

