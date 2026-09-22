import asyncio
import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
import numpy as np

app = FastAPI(title="SIH Edge Voice-AI Cloud Backend")

# Initialize Faster-Whisper Model
print("Loading Faster-Whisper ASR Model...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("Faster-Whisper Model Loaded Successfully!")

# In-Memory Storage for Live Dashboard Stats
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

# 1. DYNAMIC WEBPAGE DASHBOARD (Auto-refreshes every 2 seconds)
@app.get("/", response_class=HTMLResponse)
def home():
    log_items = "".join([f"<li>{log}</li>" for log in server_stats["logs"]])
    if not log_items:
        log_items = "<li>No active connections yet.</li>"
        
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>SIH Live Voice-AI Dashboard</title>
            <meta http-equiv="refresh" content="2">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 40px; text-align: center; }}
                .card {{ background-color: #1e293b; padding: 30px; border-radius: 12px; display: inline-block; max-width: 600px; text-align: left; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
                .badge {{ background-color: #166534; color: #4ade80; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.85rem; display: inline-block; margin-bottom: 15px; }}
                code {{ background-color: #0f172a; color: #38bdf8; padding: 4px 8px; border-radius: 6px; font-family: monospace; }}
                ul {{ background: #0f172a; padding: 15px 25px; border-radius: 8px; font-family: monospace; list-style-type: square; }}
                li {{ color: #38bdf8; margin-bottom: 6px; }}
            </style>
        </head>
        <body>
            <div class="card">
                <span class="badge">&#9679; SERVER ONLINE</span>
                <h2 style="margin-top:0;">SIH Edge Voice-AI Cloud Backend</h2>
                <p><b>Total Audio Streams Processed:</b> {server_stats['total_requests']}</p>
                <p><b>Latest Transcription:</b> <code>{server_stats['last_transcript']}</code></p>
                <h3>Recent Server Activity Logs:</h3>
                <ul>{log_items}</ul>
            </div>
        </body>
    </html>
    """

# 2. WEBSOCKET ENDPOINT FOR AUDIO STREAMING
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