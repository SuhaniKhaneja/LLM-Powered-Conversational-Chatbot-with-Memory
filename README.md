# LLM-Powered Conversational Chatbot with Session-Based Memory

## 🚀 Overview
This project is a conversational AI chatbot built using a transformer-based language model (BlenderBot) with session-based multi-turn memory. It allows users to interact in a continuous conversation via a REST API.

---

## ✨ Features
- Multi-turn conversation support
- Session-based memory handling
- Context-aware responses
- REST API using FastAPI
- Lightweight and deployable

---

## 🧠 Tech Stack
- Python
- FastAPI
- HuggingFace Transformers
- PyTorch

---

## ⚙️ How It Works
User input is stored in a session and passed along with previous conversation history to the model. The chatbot generates responses based on recent context, enabling multi-turn interactions.

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
