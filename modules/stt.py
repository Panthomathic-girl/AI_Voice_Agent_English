# modules/stt.py
import os
import logging
import whisper

def load_whisper_model(model_name: str = "tiny"):
    """Preload Whisper model."""
    try:
        model = whisper.load_model(model_name)
        return model
    except Exception as e:
        logging.error(f"Error loading Whisper model: {str(e)}")
        return None

def speech_to_text(audio_path: str, mode: str = "local", model=None) -> str:
    """Convert audio file to text using Whisper."""
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found: {audio_path}")
        return ""
    
    if mode == "local":
        try:
            if model is None:
                logging.info(f"Loading Whisper model 'tiny' for {audio_path}")
                model = whisper.load_model("tiny")
            result = model.transcribe(audio_path, language="en")
            text = result["text"].strip()
            logging.info(f"Local STT: '{text}'")
            return text
        except Exception as e:
            logging.error(f"Local STT error: {str(e)}")
            return ""
    elif mode == "groq":
        # ... (unchanged Groq STT code)
        pass
    else:
        logging.error(f"Invalid STT mode: {mode}")
        return ""