# modules/tts.py
from gtts import gTTS
from TTS.api import TTS
import pygame
import time
import os
import subprocess
import edge_tts
import asyncio
from app.config import TEMP_DIR, COQUI_MODEL_NAME, KOKORO_MODEL_NAME, FALLBACK_TTS

pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Coqui/Kokoro models (loaded on demand)
coqui_model = None
kokoro_model = None

def check_espeak():
    """Check if eSpeak-ng is installed and in PATH."""
    try:
        result = subprocess.run(['espeak-ng', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"eSpeak-ng detected: {result.stdout.strip()}")
            return True
        else:
            print("eSpeak-ng detected but failed to run. Check installation.")
            return False
    except FileNotFoundError:
        print("eSpeak-ng not found in PATH. Install from https://github.com/espeak-ng/espeak-ng/releases and add to system PATH.")
        return False
    except Exception as e:
        print(f"eSpeak-ng check failed: {e}")
        return False

def load_coqui_tts():
    global coqui_model
    if coqui_model is None:
        if not check_espeak():
            raise Exception("eSpeak-ng required for Coqui TTS. Install from https://github.com/espeak-ng/espeak-ng/releases")
        print("Loading Coqui TTS model... (one-time)")
        try:
            coqui_model = TTS(model_name=COQUI_MODEL_NAME, progress_bar=True)
            print(f"Coqui TTS '{COQUI_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            print(f"Coqui TTS Load Failed: {e}. Try clearing cache (~/.cache/tts) or changing model in config.py.")
            raise
    return coqui_model

def load_kokoro_tts():
    global kokoro_model
    if kokoro_model is None:
        if not check_espeak():
            raise Exception("eSpeak-ng required for Kokoro TTS. Install from https://github.com/espeak-ng/espeak-ng/releases")
        print("Loading Kokoro TTS model... (one-time)")
        try:
            kokoro_model = TTS(model_name=KOKORO_MODEL_NAME, progress_bar=True)
            print(f"Kokoro TTS '{KOKORO_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            print(f"Kokoro TTS Load Failed: {e}. Try clearing cache (~/.cache/tts) or changing model in config.py.")
            raise
    return kokoro_model

def gtts_text_to_speech(text):
    if not text:
        return None
    mp3_file = os.path.join(TEMP_DIR, f"out_{int(time.time()*1000)}.mp3")
    try:
        tts = gTTS(text, lang='en')
        tts.save(mp3_file)
        print(f"gTTS generated: {mp3_file}")
        return mp3_file
    except Exception as e:
        print(f"gTTS Failed: {e}")
        return None

def coqui_text_to_speech(text):
    if not text:
        return None
    wav_file = os.path.join(TEMP_DIR, f"out_{int(time.time()*1000)}.wav")
    try:
        model = load_coqui_tts()
        model.tts_to_file(text=text, file_path=wav_file)
        print(f"Coqui TTS generated: {wav_file}")
        return wav_file
    except Exception as e:
        print(f"Coqui TTS Failed: {e}")
        return None

def kokoro_text_to_speech(text):
    if not text:
        return None
    wav_file = os.path.join(TEMP_DIR, f"out_{int(time.time()*1000)}.wav")
    try:
        model = load_kokoro_tts()
        model.tts_to_file(text=text, file_path=wav_file)
        print(f"Kokoro TTS generated: {wav_file}")
        return wav_file
    except Exception as e:
        print(f"Kokoro TTS Failed: {e}. Try clearing cache (~/.cache/tts) or changing model in config.py.")
        return None

async def edge_text_to_speech(text):
    if not text:
        return None
    mp3_file = os.path.join(TEMP_DIR, f"out_{int(time.time()*1000)}.mp3")
    try:
        communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
        await communicate.save(mp3_file)
        print(f"Edge TTS generated: {mp3_file}")
        return mp3_file
    except Exception as e:
        print(f"Edge TTS Failed: {e}")
        return None

async def text_to_speech(text, mode="gtts"):
    if mode == "gtts":
        return gtts_text_to_speech(text)
    elif mode == "coqui":
        return coqui_text_to_speech(text)
    elif mode == "kokoro":
        file = kokoro_text_to_speech(text)
        if file is None and FALLBACK_TTS != "kokoro":
            print(f"Falling back to {FALLBACK_TTS.upper()} due to Kokoro failure (check eSpeak-ng: https://github.com/espeak-ng/espeak-ng/releases, or clear cache: ~/.cache/tts)")
            return await text_to_speech(text, mode=FALLBACK_TTS)
        return file
    elif mode == "edge":
        return await edge_text_to_speech(text)
    else:
        raise ValueError("Invalid TTS mode. Choose 'gtts', 'coqui', 'kokoro', or 'edge'.")

def play_audio(file_path):
    if not file_path or not os.path.exists(file_path):
        print("No audio file to play.")
        return
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        print("Speaking...")
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"Playback Error: {e}")