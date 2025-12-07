# Chatbot Backend

A Flask-based chatbot backend that supports multiple LLM models including Hugging Face models (Llama, Mistral) and Groq API.

## Features

- Support for multiple model backends:
  - **Hugging Face Transformers**: Run models locally (Llama 2, Llama 3, Mistral, etc.)
  - **Groq API**: Fast inference using Groq's API
- Chat history management per session
- RESTful API endpoints
- CORS enabled for frontend integration
- Model caching for efficient resource usage

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables (Optional)

For models that require authentication or API keys:

```bash
# For Hugging Face models that require authentication (e.g., Llama models)
export HUGGINGFACE_TOKEN="your_huggingface_token"

# For Groq API models
export GROQ_API_KEY="your_groq_api_key"
```

You can get these tokens from:
- Hugging Face: https://huggingface.co/settings/tokens
- Groq: https://console.groq.com/keys

### 3. Run the Server

```bash
python app.py
```

The server will start on `http://localhost:5000` by default.

## API Endpoints

### Health Check
```
GET /health
```

### Get Available Models
```
GET /models
```

Returns a list of available models and their configurations.

### Chat
```
POST /chat
Content-Type: application/json

{
  "message": "Hello!",
  "model": "mistral-7b",
  "session_id": "user_session_1"
}
```

Response:
```json
{
  "response": "Hello! How can I assist you today?",
  "session_id": "user_session_1",
  "model": "mistral-7b"
}
```

### Get Chat History
```
GET /chat/history?session_id=user_session_1
```

### Clear Chat History
```
POST /chat/clear
Content-Type: application/json

{
  "session_id": "user_session_1"
}
```

## Available Models

### Hugging Face Models (Local)
- `mistral-7b`: Mistral 7B Instruct (no auth required)
- `llama-2-7b`: Llama 2 7B Chat (requires HUGGINGFACE_TOKEN)
- `llama-3-8b`: Llama 3 8B Instruct (requires HUGGINGFACE_TOKEN)
- `llama-3-70b`: Llama 3 70B Instruct (requires HUGGINGFACE_TOKEN)

### Groq API Models (Fast Inference)
- `groq-llama-3-8b`: Llama 3 8B via Groq
- `groq-llama-3-70b`: Llama 3 70B via Groq
- `groq-mixtral`: Mixtral 8x7B via Groq

**Note**: Groq models require `GROQ_API_KEY` environment variable.

## Testing

Run the test script to verify everything works:

```bash
python test_chatbot.py
```

Make sure the server is running before executing the test script.

## Usage Examples

### Using cURL

```bash
# Send a chat message
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is Python?",
    "model": "mistral-7b",
    "session_id": "my_session"
  }'
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:5000/chat",
    json={
        "message": "Hello!",
        "model": "mistral-7b",
        "session_id": "test_session"
    }
)

print(response.json()["response"])
```

## Notes

- Models are loaded lazily (only when first used) and cached in memory
- Chat history is stored in memory (will be lost on server restart)
- For production, consider using a database for chat history
- GPU support is automatic if CUDA is available
- CPU inference is supported but will be slower

## Frontend Integration

The API is ready for frontend integration. All endpoints support CORS, so you can call them from any frontend application.

Example frontend code:
```javascript
const response = await fetch('http://localhost:5000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: userMessage,
    model: selectedModel,
    session_id: sessionId
  })
});

const data = await response.json();
console.log(data.response);
```

