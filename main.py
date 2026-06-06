import os
import re
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
import requests

app = FastAPI(title="RYU Core API", version="10.5.4")

RYU_MASTER_KEY = os.getenv("RYU_API_KEY", "ryu_internal_secure_core_v10")
api_key_header = APIKeyHeader(name="X-RYU-Token", auto_error=True)

def verify_token(api_key: str = Depends(api_key_header)):
    if api_key != RYU_MASTER_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    return api_key

class ChatMessage(BaseModel):
    role: str
    content: str

class Payload(BaseModel):
    messages: List[ChatMessage]
    model_preference: str = "gemini" 

def sanitize_output(text: str) -> str:
    if not text:
        return ""
    text = text.replace("*", "").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    action_tags = ["sigh", "sighs", "deadpan", "chuckle", "smirk", "pause", "mutter", "shrug"]
    for tag in action_tags:
        text = re.sub(rf'\b{tag}\b\s*,?\s*', '', text, flags=re.IGNORECASE)
    
    text = text.strip()
    if text and not text.startswith("```") and text[0].islower():
        text = text[0].upper() + text[1:]
    return text

def route_to_gemini_native(messages: list) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing API Key.")
    
    # Extract the system prompt cleanly if it exists, or provide a default
    system_instruction = "You are a helpful assistant."
    contents = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_instruction = msg["content"]
        else:
            # Native Gemini endpoints only accept 'user' and 'model'
            role_map = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role_map,
                "parts": [{"text": msg["content"]}]
            })
            
    # If the history slice doesn't have a user message yet, append an empty one to satisfy API specs
    if not contents:
        contents.append({"role": "user", "parts": [{"text": "Hello"}]})
        
    url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=){api_key}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "temperature": 0.6,
            "maxOutputTokens": 400
        }
    }
    
    response = requests.post(url, json=payload, headers=headers, timeout=15)
    if response.status_code == 200:
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise HTTPException(status_code=500, detail="Parsing failure.")
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/v1/chat/generate", dependencies=[Depends(verify_token)])
async def generate_response(payload: Payload):
    history = [msg.dict() for msg in payload.messages]
    try:
        raw_response = route_to_gemini_native(history)
        cleaned_response = sanitize_output(raw_response)
        return {"status": "success", "response": cleaned_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
