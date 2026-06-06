import os
import re
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict
import requests

app = FastAPI(title="RYU Core API", version="10.5.3")

# This secures your private API so random people can't use your credits
RYU_MASTER_KEY = os.getenv("RYU_API_KEY", "ryu_internal_secure_core_v10")
api_key_header = APIKeyHeader(name="X-RYU-Token", auto_error=True)

def verify_token(api_key: str = Depends(api_key_header)):
    if api_key != RYU_MASTER_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized RYU interface access.")
    return api_key

class ChatMessage(BaseModel):
    role: str
    content: str

class Payload(BaseModel):
    messages: List[ChatMessage]
    model_preference: str = "gemini" 

# =========================
# INTERNAL CLEANER ENGINE
# =========================
def sanitize_output(text: str) -> str:
    """Strips conversational noise and action tags before returning text."""
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

# =========================
# NATIVE GOOGLE ROUTING LAYER
# =========================
def route_to_gemini_native(messages: list) -> str:
    """Direct connection to official Google API Studio endpoint."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Native Gemini credentials missing from server env.")
    
    # Format message structure to match Google's native developer schema
    contents = []
    for msg in messages:
        # Maps user/system to 'user', assistant to 'model'
        role = "user" if msg["role"] in ["user", "system"] else "model"
        contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
        
    url = f"[https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=](https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=){api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents, "generationConfig": {"temperature": 0.6}}
    
    response = requests.post(url, json=payload, headers=headers, timeout=15)
    if response.status_code == 200:
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise HTTPException(status_code=500, detail="Unexpected JSON structure from Google API.")
    else:
        raise HTTPException(status_code=response.status_code, detail=f"Google Native API error: {response.text}")

# =========================
# CORE GENERATE ENDPOINT
# =========================
@app.post("/v1/chat/generate", dependencies=[Depends(verify_token)])
async def generate_response(payload: Payload):
    history = [msg.dict() for msg in payload.messages]
    
    try:
        # Defaults straight to high-limit official Google API pipelines
        raw_response = route_to_gemini_native(history)
        cleaned_response = sanitize_output(raw_response)
        return {"status": "success", "response": cleaned_response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
