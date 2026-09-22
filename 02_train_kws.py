import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORD_DIR = os.path.join(BASE_DIR, "dataset", "keyword")
NOISE_DIR = os.path.join(BASE_DIR, "dataset", "noise")

# 2. Feature Extraction Function (Spectrogram Creation)
def extract_spectrogram(file_path):
    # Load audio at 16kHz
    y, sr = librosa.load(file_path, sr=16000, duration=1.0)
    # Pad or truncate to exactly 16000 samples (1 second)
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    else:
        y = y[:16000]
    
    # Generate Mel-Spectrogram (50 time steps x 40 mel bins)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=320, n_fft=480)
    log_spec = librosa.power_to_db(spec, ref=np.max)
    # Standardize shape to (1, 50, 40)
    return log_spec.T[:50, :40][np.newaxis, :, :]

print("Extracting audio features from dataset...")
X, y = [], []

# Load Keyword Audio Files (Class 1: Hey Mitchell)
keyword_files = glob.glob(os.path.join(KEYWORD_DIR, "*.*"))
for f in keyword_files:
    try:
        X.append(extract_spectrogram(f))
        y.append(1) # Label 1 = Keyword
    except Exception as e:
        pass

# Load Noise Audio Files (Class 0: Noise/Background)
noise_files = glob.glob(os.path.join(NOISE_DIR, "*.*"))
for f in noise_files:
    try:
        X.append(extract_spectrogram(f))
        y.append(0) # Label 0 = Noise
    except Exception as e:
        pass

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print(f"Dataset Loaded: {len(X)} samples with shape {X.shape}")

# 3. Ultra-Lightweight DS-CNN Architecture (Fits in <256KB RAM)
class DSCNN(nn.Module):
    def __init__(self):
        super(DSCNN, self).__init__()
        self.features = nn.Sequential(
            # Standard Conv
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Depthwise Separable Block
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 2) # 2 Classes: Noise vs Keyword

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 4. Train Model
model = DSCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

inputs = torch.tensor(X)
labels = torch.tensor(y)

print("Training KWS Model...")
for epoch in range(15): # 15 Fast Training Epochs
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/15], Loss: {loss.item():.4f}")

# Save PyTorch Model Weights
torch.save(model.state_dict(), "kws_model.pth")
print("Model trained and saved as 'kws_model.pth' successfully!")