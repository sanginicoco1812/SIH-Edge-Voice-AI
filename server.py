import asyncio
import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import numpy as np

app = FastAPI(title="SIH Edge Voice-AI Cloud Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Faster-Whisper ASR Model...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("Faster-Whisper Model Loaded Successfully!")

server_stats = {
    "total_requests": 0,
    "last_transcript": "None yet",
    "logs": []
}

def log_event(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    server_stats["logs"].insert(0, entry)
    if len(server_stats["logs"]) > 10:
        server_stats["logs"].pop()

# Serve static index.html from root route
@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>index.html file not found in root directory</h3>"

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "total_requests": server_stats["total_requests"],
        "last_transcript": server_stats["last_transcript"],
        "logs": server_stats["logs"]
    }

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log_event("Client connected over WebSocket")
    audio_bytes = bytearray()
    
    try:
        while True:
            chunk = await websocket.receive_bytes()
            audio_bytes.extend(chunk)
    except WebSocketDisconnect:
        server_stats["total_requests"] += 1
        if len(audio_bytes) > 0:
            pcm_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if np.max(np.abs(pcm_data)) < 0.01:
                transcript = "silence"
            else:
                segments, _ = whisper_model.transcribe(pcm_data, beam_size=1)
                transcript = "".join([segment.text for segment in segments]).strip() or "silence"

            server_stats["last_transcript"] = transcript
            log_event(f"Transcribed: '{transcript}'")
            
            try:
                await websocket.send_json({"status": "received", "transcript": transcript})
            except Exception:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)