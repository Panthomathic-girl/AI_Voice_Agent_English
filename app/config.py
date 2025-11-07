# # config.py
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # === API KEYS ===
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # New: LiveKit
# LIVEKIT_URL = os.getenv("LIVEKIT_URL")
# LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
# LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# # === AUDIO SETTINGS ===
# SAMPLE_RATE = 16000
# RECORD_SECONDS = 5
# WHISPER_MODEL_NAME = "base"  # For local Whisper: tiny / base / small

# # === TTS SETTINGS ===
# COQUI_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"  # Coqui: voice cloning, multilingual
# KOKORO_MODEL_NAME = "tts_models/en/ljspeech/vits"  # Kokoro: realistic voices
# FALLBACK_TTS = "gtts"  # Fallback if Kokoro/Coqui fails

# # === GEMINI MODEL ===
# GEMINI_MODEL = "gemini-2.5-flash"  # Fast & cheap
# GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-09-2025"  # For Live API

# # === GROQ MODEL ===
# GROQ_MODEL = "llama-3.1-8b-instant"

# # === PATHS ===
# TEMP_DIR = "temp_audio"
# LOG_DIR = "logs"
# os.makedirs(TEMP_DIR, exist_ok=True)
# os.makedirs(LOG_DIR, exist_ok=True)

# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# === API KEYS ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "your-gemini-api-key-here"
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "your-groq-api-key-here"

# === AUDIO SETTINGS ===
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
WHISPER_MODEL_NAME = "base"  # For local Whisper: tiny / base / small

# === TTS SETTINGS ===
COQUI_MODEL_NAME = "tts_models/en/ljspeech/tacotron2-DDC"  # Coqui: voice cloning, multilingual
KOKORO_MODEL_NAME = "tts_models/en/ljspeech/vits"  # Kokoro: realistic voices
FALLBACK_TTS = "gtts"  # Fallback if Kokoro/Coqui fails

# === GEMINI MODEL ===
GEMINI_MODEL = "gemini-2.5-flash"  # Fast & cheap

# === GROQ MODEL ===
GROQ_MODEL = "llama-3.1-8b-instant"

# === PATHS ===
TEMP_DIR = "temp_audio"
LOG_DIR = "logs"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)