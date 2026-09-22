import os
import subprocess
import numpy as np
import wave

# Setup Folder Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORD_DIR = os.path.join(BASE_DIR, "dataset", "keyword")
NOISE_DIR = os.path.join(BASE_DIR, "dataset", "noise")
UNKNOWN_DIR = os.path.join(BASE_DIR, "dataset", "unknown")

# Safely recreate dataset structure
for folder in [KEYWORD_DIR, NOISE_DIR, UNKNOWN_DIR]:
    os.makedirs(folder, exist_ok=True)

# 1. Custom Wake-Word
WAKE_WORD = "Hey Mitchell"

print(f"Generating synthetic custom keyword audio for: '{WAKE_WORD}'...")

# Mac default TTS speeds
rates = [140, 160, 180, 200, 220]

count = 0
for rate in rates:
    output_aiff = os.path.join(KEYWORD_DIR, f"keyword_{count}.aiff")
    cmd = f'say -r {rate} -o "{output_aiff}" "{WAKE_WORD}"'
    
    res = subprocess.run(cmd, shell=True)
    if res.returncode == 0:
        count += 1

print(f"Generated {count} clean samples in '{KEYWORD_DIR}'!")

# 2. Generate Synthetic Ambient Background Noise Clips using standard wave library
print("Generating synthetic noise clips...")
for i in range(20):
    noise_float = np.random.normal(0, 0.05, 16000)
    noise_int16 = (noise_float * 32767).astype(np.int16)
    
    noise_path = os.path.join(NOISE_DIR, f"noise_{i}.wav")
    
    # Built-in Python WAV writer (Zero external dependency issues)
    with wave.open(noise_path, 'w') as f:
        f.setnchannels(1) # Mono
        f.setsampwidth(2) # 16-bit PCM
        f.setframerate(16000) # 16kHz
        f.writeframes(noise_int16.tobytes())

print("Dataset folders prepared successfully!")