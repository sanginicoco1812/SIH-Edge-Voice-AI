import io
import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel

app = FastAPI()

print("Loading Faster-Whisper ASR Model (tiny.en)...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("ASR Model Loaded Successfully!")

@app.websocket("/stream")
async def audio_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(">>> Edge device connected over WebSocket! <<<")
    
    audio_buffer = bytearray()
    
    try:
        while True:
            chunk = await websocket.receive_bytes()
            audio_buffer.extend(chunk)
            
            # 1 Second Audio Buffer (32000 bytes)
            if len(audio_buffer) >= 32000:
                # Convert raw PCM int16 bytes to float32 numpy array for Whisper
                pcm_data = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                
                try:
                    segments, _ = whisper_model.transcribe(pcm_data, beam_size=1)
                    text = "".join([s.text for s in segments]).strip()
                except Exception as e:
                    text = ""
                
                if text:
                    print(f"Recognized Speech: '{text}'")
                    await websocket.send_json({"status": "success", "transcript": text})
                else:
                    print("Received 1s Audio Chunk (Silence/Noise)")
                    await websocket.send_json({"status": "received", "transcript": "silence"})
                
                audio_buffer.clear()
                
    except WebSocketDisconnect:
        print("Edge device disconnected normally.")
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    import uvicorn
    print("Starting Cloud Streaming Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    