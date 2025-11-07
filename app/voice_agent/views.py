from fastapi.responses import HTMLResponse
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio
import json
import tempfile
import io
from typing import AsyncGenerator
import logging
import os
import time
from pydub import AudioSegment
from modules.stt import speech_to_text
from modules.llm import gemini_response, groq_response
from app.config import SAMPLE_RATE, WHISPER_MODEL_NAME, GROQ_API_KEY, GEMINI_API_KEY, GEMINI_MODEL, FALLBACK_TTS
from modules.tts import text_to_speech

router = APIRouter(prefix="/voice_agent", tags=["Voice Agent"])

# Templates for HTML
templates = Jinja2Templates(directory="templates")

# Voice Agent Class
class VoiceAgent:
    def __init__(self, stt_mode: str = "local", tts_mode: str = "gtts", llm_mode: str = "gemini"):
        self.stt_mode = stt_mode
        self.tts_mode = tts_mode
        self.llm_mode = llm_mode
        logging.basicConfig(level=logging.INFO)

    async def process_audio_to_text(self, audio_bytes: bytes, stt_mode: str, file_extension: str) -> str:
        """STT: Audio bytes to text."""
        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            # Convert WebM to WAV if necessary
            if file_extension.lower() == "webm":
                wav_path = tmp_path.replace(".webm", ".wav")
                audio = AudioSegment.from_file(tmp_path, format="webm")
                audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
                audio.export(wav_path, format="wav")
                tmp_path = wav_path
            text = speech_to_text(tmp_path, mode=stt_mode)
            logging.info(f"STT Result: {text}")
            return text if text else "[no speech]"
        finally:
            os.unlink(tmp_path)
            if file_extension.lower() == "webm" and os.path.exists(wav_path):
                os.unlink(wav_path)

    async def generate_response(self, text: str, llm_mode: str) -> str:
        """LLM: Text to response using Gemini or Groq."""
        if not text or text == "[no speech]":
            return "I didn't hear anything. Please speak again."
        if "bye" in text.lower() or "exit" in text.lower():
            return "Goodbye! Have a great day."
        prompt = f"""Answer the question concisely and clearly in English.

Question: {text}

Answer:"""
        if llm_mode == "gemini":
            response = gemini_response(prompt)
        elif llm_mode == "groq":
            response = groq_response(prompt)
        else:
            raise ValueError("Invalid LLM mode. Choose 'gemini' or 'groq'.")
        logging.info(f"LLM Response ({llm_mode}): {response}")
        return response

    async def stream_llm_response(self, text: str, llm_mode: str) -> AsyncGenerator[dict, None]:
        """Stream LLM response in word-sized chunks with TTFC."""
        full_response = await self.generate_response(text, llm_mode)
        words = full_response.split()
        first_chunk = True
        start_time = time.time()
        for word in words:
            chunk = word + " "
            yield {
                "type": "text_chunk",
                "chunk": chunk,
                "ttfc": time.time() - start_time if first_chunk else None
            }
            first_chunk = False
            await asyncio.sleep(0.05)  # 50ms delay for word-sized chunks
        if full_response.endswith((".", "!", "?")):
            yield {
                "type": "text_chunk",
                "chunk": full_response[-1],
                "ttfc": None
            }

    async def text_to_audio(self, text: str, tts_mode: str) -> bytes:
        """TTS: Text to audio bytes."""
        file_path = await text_to_speech(text, mode=tts_mode)
        if not file_path or not os.path.exists(file_path):
            logging.warning(f"TTS failed for {tts_mode}, falling back to {FALLBACK_TTS}")
            file_path = await text_to_speech(text, mode=FALLBACK_TTS)
        if not file_path:
            raise HTTPException(status_code=500, detail="TTS generation failed")
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        os.unlink(file_path)
        return audio_bytes

# Global agent instance
agent = VoiceAgent(stt_mode="local", tts_mode="gtts", llm_mode="gemini")

@router.get("/")
async def index(request: Request):
    """Serve HTML interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(..., description="Upload audio file (WAV/MP3/WebM)"),
    stt_mode: str = Query("local", description="STT mode: local or groq"),
    tts_mode: str = Query("gtts", description="TTS mode: gtts, coqui, kokoro, or edge"),
    llm_mode: str = Query("gemini", description="LLM mode: gemini or groq")
):
    """Upload audio, process speech-to-speech, return audio and text."""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.webm')):
        raise HTTPException(status_code=415, detail="Only WAV/MP3/WebM files are supported.")
    if stt_mode not in ["local", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid STT mode. Choose 'local' or 'groq'.")
    if tts_mode not in ["gtts", "coqui", "kokoro", "edge"]:
        raise HTTPException(status_code=400, detail="Invalid TTS mode. Choose 'gtts', 'coqui', 'kokoro', or 'edge'.")
    if llm_mode not in ["gemini", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid LLM mode. Choose 'gemini' or 'groq'.")
    contents = await file.read()
    file_extension = file.filename.split('.')[-1].lower()
    try:
        text = await agent.process_audio_to_text(contents, stt_mode=stt_mode, file_extension=file_extension)
        response_text = await agent.generate_response(text, llm_mode=llm_mode)
        audio_bytes = await agent.text_to_audio(response_text, tts_mode=tts_mode)
        return JSONResponse({
            "transcription": text,
            "response": response_text,
            "audio": audio_bytes.hex(),
            "supportMessage": {"label": "Would you like to know more?", "options": ["Record Again"]}
        })
    except Exception as e:
        logging.error(f"Upload processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/voice-stream")
async def voice_stream(
    file: UploadFile = File(..., description="Upload audio file (WAV/MP3/WebM)"),
    stt_mode: str = Query("local", description="STT mode: local or groq"),
    tts_mode: str = Query("gtts", description="TTS mode: gtts, coqui, kokoro, or edge"),
    llm_mode: str = Query("gemini", description="LLM mode: gemini or groq")
):
    """Stream voice response: STT -> streamed LLM text chunks -> TTS audio."""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.webm')):
        raise HTTPException(status_code=415, detail="Only WAV/MP3/WebM files are supported.")
    if stt_mode not in ["local", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid STT mode. Choose 'local' or 'groq'.")
    if tts_mode not in ["gtts", "coqui", "kokoro", "edge"]:
        raise HTTPException(status_code=400, detail="Invalid TTS mode. Choose 'gtts', 'coqui', 'kokoro', or 'edge'.")
    if llm_mode not in ["gemini", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid LLM mode. Choose 'gemini' or 'groq'.")
    contents = await file.read()
    file_extension = file.filename.split('.')[-1].lower()
    start_time = time.time()
    text = await agent.process_audio_to_text(contents, stt_mode=stt_mode, file_extension=file_extension)

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            yield f"data: {json.dumps({'type': 'transcription', 'text': text, 'timestamp': time.time()})}\n\n"
            full_response = ""
            async for chunk_data in agent.stream_llm_response(text, llm_mode):
                full_response += chunk_data["chunk"]
                yield f"data: {json.dumps(chunk_data)}\n\n"
            audio_bytes = await agent.text_to_audio(full_response, tts_mode=tts_mode)
            yield f"data: {json.dumps({'type': 'audio', 'data': audio_bytes.hex()})}\n\n"
            yield f"data: {json.dumps({'supportMessage': {'label': 'Would you like to know more?', 'options': ['Record Again']}})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logging.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/stream")
async def text_stream(
    query: str = Query(..., description="Text query for streaming"),
    stt_mode: str = Query("local", description="STT mode: local or groq (not used for text)"),
    tts_mode: str = Query("gtts", description="TTS mode: gtts, coqui, kokoro, or edge"),
    llm_mode: str = Query("gemini", description="LLM mode: gemini or groq")
):
    """Stream text response: streamed LLM text chunks -> TTS audio."""
    if llm_mode not in ["gemini", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid LLM mode. Choose 'gemini' or 'groq'.")
    if tts_mode not in ["gtts", "coqui", "kokoro", "edge"]:
        raise HTTPException(status_code=400, detail="Invalid TTS mode. Choose 'gtts', 'coqui', 'kokoro', or 'edge'.")
    text = query
    start_time = time.time()

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            full_response = ""
            async for chunk_data in agent.stream_llm_response(text, llm_mode):
                full_response += chunk_data["chunk"]
                yield f"data: {json.dumps(chunk_data)}\n\n"
            audio_bytes = await agent.text_to_audio(full_response, tts_mode=tts_mode)
            yield f"data: {json.dumps({'type': 'audio', 'data': audio_bytes.hex()})}\n\n"
            yield f"data: {json.dumps({'supportMessage': {'label': 'Would you like to know more?', 'options': ['Record Again']}})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logging.error(f"Stream error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")