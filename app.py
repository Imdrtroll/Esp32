# app.py
from fastapi import FastAPI, HTTPException, Query
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="ESP32 AI API")

MODEL_NAME = "distilgpt2"  # CPU friendly
device = "cpu"  # Koyeb free tier is CPU only

print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
print("Model loaded!")

@app.get("/chat")
async def chat(prompt: str = Query(..., min_length=1)):
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=100)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"response": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
