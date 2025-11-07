# modules/llm.py
import google.generativeai as genai
from groq import Groq
from app.config import GEMINI_API_KEY, GROQ_API_KEY, GEMINI_MODEL, GROQ_MODEL
import logging

logging.basicConfig(level=logging.INFO)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

groq_client = Groq(api_key=GROQ_API_KEY)

def gemini_response(user_text):
    if not user_text.strip():
        return "I didn't hear anything. Please speak again."
    
    try:
        prompt = f"You are a helpful voice assistant. Respond naturally and concisely.\nUser: {user_text}\nAssistant:"
        response = gemini_model.generate_content(prompt)
        reply = response.text.strip()
        print(f"Gemini: {reply}")
        return reply
    except Exception as e:
        print(f"Gemini Error: {e}")
        logging.error(f"Gemini detailed error: {e}")
        return "Sorry, I couldn't process that."

def groq_response(user_text):
    if not user_text.strip():
        return "I didn't hear anything. Please speak again."
    
    try:
        prompt = f"You are a helpful voice assistant. Respond naturally and concisely.\nUser: {user_text}\nAssistant:"
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,  # Use config model (llama3.1-8b-instant)
            temperature=0.7,
            max_tokens=150
        )
        reply = response.choices[0].message.content.strip()
        print(f"Groq: {reply}")
        return reply
    except Exception as e:
        print(f"Groq Error: {e}")
        logging.error(f"Groq detailed error: {e}")
        return "Sorry, I couldn't process that."

# Default export (for backward compatibility)
gemini_response