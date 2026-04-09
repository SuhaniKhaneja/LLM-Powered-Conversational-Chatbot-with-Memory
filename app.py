import re
import warnings
from typing import Dict, Optional
from uuid import uuid4

warnings.filterwarnings("ignore")

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model config
MODEL_NAME = "facebook/blenderbot-400M-distill"
MAX_HISTORY_TURNS = 5

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
model.eval()

# Memory store
sessions: Dict[str, list] = {}

def get_or_create_session(session_id: Optional[str]):
    if session_id is None or session_id not in sessions:
        session_id = str(uuid4())
        sessions[session_id] = []
    return session_id, sessions[session_id]

# Response generator
def generate_response(user_input: str, history: list) -> str:
    history.append(f"User: {user_input}")

    conversation = "You are a helpful chatbot.\n"
    for turn in history[-MAX_HISTORY_TURNS:]:
        conversation += turn + "\n"
    conversation += "Bot:"

    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.6,
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.3
        )

    reply = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = reply.split("Bot:")[-1].strip()

    history.append(f"Bot: {reply}")

    return reply

# API schemas
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    bot_response: str

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        session_id, history = get_or_create_session(request.session_id)
        reply = generate_response(request.message, history)

        return ChatResponse(
            session_id=session_id,
            bot_response=reply
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
