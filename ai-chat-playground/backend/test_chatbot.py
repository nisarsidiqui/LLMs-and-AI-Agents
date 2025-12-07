"""
Simple test script to test the chatbot API
Run this after starting the Flask server with: python app.py
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}")
    return response.status_code == 200

def test_get_models():
    """Test getting available models"""
    response = requests.get(f"{BASE_URL}/models")
    print(f"\nAvailable models:")
    models = response.json()
    for model_name in models["models"]:
        print(f"  - {model_name}")
    return response.status_code == 200

def test_chat(message: str, model: str = "mistral-7b", session_id: str = "test_session"):
    """Test chat endpoint"""
    payload = {
        "message": message,
        "model": model,
        "session_id": session_id
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nUser: {message}")
        print(f"Assistant ({model}): {data['response']}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

def test_chat_history(session_id: str = "test_session"):
    """Test getting chat history"""
    response = requests.get(f"{BASE_URL}/chat/history?session_id={session_id}")
    if response.status_code == 200:
        data = response.json()
        print(f"\nChat history for session '{session_id}':")
        for msg in data["history"]:
            role = msg["role"]
            content = msg["content"]
            print(f"  {role.capitalize()}: {content[:100]}..." if len(content) > 100 else f"  {role.capitalize()}: {content}")
        return True
    else:
        print(f"Error: {response.json()}")
        return False

if __name__ == "__main__":
    print("Testing Chatbot API")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("Server is not running. Please start it with: python app.py")
        exit(1)
    
    # Test getting models
    test_get_models()
    
    # Test chat
    print("\n" + "=" * 50)
    print("Testing Chat Functionality")
    print("=" * 50)
    
    # Use a model that doesn't require authentication by default
    model = "mistral-7b"
    
    test_chat("Hello! How are you?", model=model)
    test_chat("What is the capital of France?", model=model)
    test_chat("Can you explain what machine learning is?", model=model)
    
    # Test chat history
    test_chat_history()

