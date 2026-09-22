import asyncio
import websockets
import ssl

async def test_stream():
    uri = "wss://cold-lions-admire.loca.lt/stream"
    print("Connecting to Edge Voice-AI Cloud Server...")
    
    # Disable strict SSL verification for localtunnel self-signed certificate
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print("Connected successfully over public internet tunnel!")
            
            # Send 1 second of PCM dummy audio (32000 bytes)
            dummy_audio = b'\x00' * 32000
            
            # Stream in 20ms chunks
            chunk_size = 640
            for i in range(0, len(dummy_audio), chunk_size):
                chunk = dummy_audio[i:i+chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(0.01)
            
            print("Audio stream sent completely! Waiting for server response...")
            
            response = await websocket.recv()
            print(f"Server Response: {response}")
            
    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())