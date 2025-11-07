# app/comparison/views.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import json
import tempfile
import os
import time
import logging
from typing import AsyncGenerator
from pydub import AudioSegment
from modules.stt import speech_to_text
from modules.llm import gemini_response, groq_response
from modules.tts import text_to_speech
from app.config import SAMPLE_RATE, FALLBACK_TTS, WHISPER_MODEL_NAME

router = APIRouter(prefix="/compare", tags=["Comparison"])

# Preload Whisper model to reduce STT latency
logging.basicConfig(level=logging.INFO)
logging.info("Preloading Local Whisper model...")
try:
    from modules.stt import load_whisper_model
    whisper_model = load_whisper_model(WHISPER_MODEL_NAME)  # Use configured model (e.g., 'tiny')
    logging.info(f"Local Whisper '{WHISPER_MODEL_NAME}' loaded.")
except Exception as e:
    logging.error(f"Failed to preload Whisper model: {str(e)}")
    whisper_model = None

class VoiceAgent:
    def __init__(self, stt_mode: str = "local", tts_mode: str = "gtts", llm_mode: str = "gemini"):
        self.stt_mode = stt_mode
        self.tts_mode = tts_mode
        self.llm_mode = llm_mode

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
            text = speech_to_text(tmp_path, mode=stt_mode, model=whisper_model if stt_mode == "local" else None)
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
        """Stream LLM response in word-sized chunks for TTFC comparison."""
        full_response = await self.generate_response(text, llm_mode)
        words = full_response.split()
        first_chunk = True
        start_time = time.time()
        for word in words:
            chunk = word + " "
            yield {
                "chunk": chunk,
                "ttfc": time.time() - start_time if first_chunk else None
            }
            first_chunk = False
            await asyncio.sleep(0.05)  # 50ms delay per word
        if full_response.endswith((".", "!", "?")):
            yield {
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

@router.websocket("/ws-stream")
async def websocket_stream(websocket: WebSocket, stt_mode: str = "local", tts_mode: str = "gtts", llm_mode: str = "gemini"):
    """WebSocket streaming for comparison (measures TTFC from connect to first chunk)."""
    if stt_mode not in ["local", "groq"]:
        await websocket.close(code=1008, reason="Invalid STT mode.")
        return
    if tts_mode not in ["gtts", "coqui", "kokoro", "edge"]:
        await websocket.close(code=1008, reason="Invalid TTS mode.")
        return
    if llm_mode not in ["gemini", "groq"]:
        await websocket.close(code=1008, reason="Invalid LLM mode.")
        return
    await websocket.accept()
    try:
        data = await websocket.receive_bytes()
        if not data:
            logging.error("No audio data received")
            await websocket.send_text(json.dumps({"type": "error", "message": "No audio data received"}))
            await websocket.close(code=1003, reason="No audio data")
            return
        logging.info(f"Received audio data: {len(data)} bytes")
        start_time = time.time()
        text = await agent.process_audio_to_text(data, stt_mode=stt_mode, file_extension="webm")
        
        await websocket.send_text(json.dumps({"type": "transcription", "text": text, "timestamp": time.time()}))
        
        full_response = ""
        async for chunk_data in agent.stream_llm_response(text, llm_mode):
            full_response += chunk_data["chunk"]
            await websocket.send_text(json.dumps({
                "type": "text_chunk",
                "chunk": chunk_data["chunk"],
                "ttfc": chunk_data["ttfc"]
            }))
        
        audio_bytes = await agent.text_to_audio(full_response, tts_mode=tts_mode)
        await websocket.send_bytes(audio_bytes)
        
        await websocket.send_text(json.dumps({
            "type": "complete",
            "supportMessage": {"label": "Comparison Complete", "options": ["Test Again"]}
        }))
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
    finally:
        await websocket.close()

@router.post("/http-stream")
async def http_stream(
    file: UploadFile = File(...),
    stt_mode: str = Query("local"),
    tts_mode: str = Query("gtts"),
    llm_mode: str = Query("gemini")
):
    """HTTP Streaming for comparison (SSE with TTFC)."""
    if not file.filename.lower().endswith(('.wav', '.mp3', '.webm')):
        raise HTTPException(status_code=415, detail="Only WAV/MP3/WebM supported.")
    if stt_mode not in ["local", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid STT mode.")
    if tts_mode not in ["gtts", "coqui", "kokoro", "edge"]:
        raise HTTPException(status_code=400, detail="Invalid TTS mode.")
    if llm_mode not in ["gemini", "groq"]:
        raise HTTPException(status_code=400, detail="Invalid LLM mode.")
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
                yield f"data: {json.dumps({'type': 'text_chunk', 'chunk': chunk_data['chunk'], 'ttfc': chunk_data['ttfc']})}\n\n"
            audio_bytes = await agent.text_to_audio(full_response, tts_mode=tts_mode)
            yield f"data: {json.dumps({'type': 'audio', 'data': audio_bytes.hex()})}\n\n"
            yield f"data: {json.dumps({'supportMessage': {'label': 'HTTP Stream Complete', 'options': ['Test Again']}})}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logging.error(f"HTTP Stream error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/compare")
async def compare_latencies(stt_mode: str = "local", llm_mode: str = "gemini", tts_mode: str = "gtts"):
    """Simple endpoint to compare simulated TTFC (no audio needed; uses fixed query for benchmark)."""
    if tts_mode not in ["gtts", "coqui", "kokoro", "edge"]:
        raise HTTPException(status_code=400, detail="Invalid TTS mode.")
    fixed_query = "Who is the Prime Minister of India?"
    agent_temp = VoiceAgent(stt_mode=stt_mode, llm_mode=llm_mode, tts_mode=tts_mode)
    
    ws_start = time.time()
    full_ws = await agent_temp.generate_response(fixed_query, llm_mode)
    ws_ttf_sim = (time.time() - ws_start) / len(full_ws.split()) + 0.02  # Per word + WebSocket overhead
    
    http_start = time.time()
    full_http = await agent_temp.generate_response(fixed_query, llm_mode)
    http_ttf_sim = (time.time() - http_start) / len(full_http.split())  # Per word

    return JSONResponse({
        "query": fixed_query,
        "websocket": {"ttfc_ms": round(ws_ttf_sim * 1000, 2), "notes": "Includes connection overhead"},
        "http_streaming": {"ttfc_ms": round(http_ttf_sim * 1000, 2), "notes": "Faster for initial chunk"},
        "winner": "HTTP Streaming" if http_ttf_sim < ws_ttf_sim else "WebSocket",
        "full_response_time_ms": round((time.time() - http_start) * 1000, 2)
    })