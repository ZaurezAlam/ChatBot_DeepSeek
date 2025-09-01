# main.py - Simple AI Chat API Server
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. SETUP: Initialize FastAPI app with CORS for web requests
app = FastAPI(title="AI Chat API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 2. LOAD AI MODEL: Load DeepSeek model and tokenizer
MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, load_in_8bit=True, device_map="auto")
model.eval()

# 3. DATA STORAGE: In-memory chat sessions
sessions = {}  # {session_id: [{"role": "user/assistant", "text": "message"}]}

# 4. REQUEST/RESPONSE MODELS: Define API input/output structure
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

# 5. PROMPT BUILDER: Convert chat history to model format
def build_prompt(history):
    """Convert chat messages to ChatML format that DeepSeek understands"""
    prompt = "<|system|>\nYou are a helpful assistant.\n<|end|>\n"
    for msg in history:
        if msg["role"] == "user":
            prompt += f"<|user|>\n{msg['text']}\n<|end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant|>\n{msg['text']}\n<|end|>\n"
    prompt += "<|assistant|>\n"  # Signal for model to respond
    return prompt

# 6. MAIN CHAT ENDPOINT: Process user messages and generate AI responses
@app.post("/chat")
def chat(req: ChatRequest):
    # Get or create chat session
    session = sessions.setdefault(req.session_id, [])
    
    # Add user message to history
    session.append({"role": "user", "text": req.message})
    
    # Keep only last 10 messages to prevent memory issues
    if len(session) > 10:
        session = session[-10:]
        sessions[req.session_id] = session
    
    # Convert chat to prompt format
    prompt = build_prompt(session)
    
    # Generate AI response
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    
    # Extract just the new response
    full_text = tokenizer.decode(output[0], skip_special_tokens=False)
    response_text = full_text.split("<|assistant|>")[-1].split("<|end|>")[0].strip()
    
    # Save AI response to session
    session.append({"role": "assistant", "text": response_text})
    
    return ChatResponse(response=response_text, session_id=req.session_id)

# 7. UTILITY ENDPOINTS: Helper functions for session management
@app.get("/")
def status():
    return {"status": "AI Chat Server Running", "model": MODEL_NAME}

@app.delete("/clear/{session_id}")
def clear_chat(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": f"Session {session_id} cleared"}

# 8. RUN SERVER: Start the API server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)