import asyncio
import websockets

URI = "ws://127.0.0.1:8000/stream"

async def test_stream():
    print("Connecting to Edge Voice-AI Cloud Server...")
    try:
        websocket = await websockets.connect(URI)
        print("Connected! Sending audio chunk...")
        
        # Send 1 second of dummy PCM 16-bit 16kHz silence
        dummy_audio = b'\x00' * 32000
        await websocket.send(dummy_audio)
        
        # Close connection to trigger server-side transcription processing
        await websocket.close()
        
    except websockets.exceptions.ConnectionClosedOK:
        print("Stream finished & connection closed cleanly by server!")
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())