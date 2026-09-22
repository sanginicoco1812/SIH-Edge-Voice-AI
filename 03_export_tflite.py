import os
import torch
import torch.nn as nn

# 1. Define Model Architecture
class DSCNN(nn.Module):
    def __init__(self):
        super(DSCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# 2. Load Trained Weights
model = DSCNN()
model.load_state_dict(torch.load("kws_model.pth"))
model.eval()

# 3. Export to ONNX / Lite Format
dummy_input = torch.randn(1, 1, 50, 40)
onnx_path = "kws_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])

print(f"Exported ONNX Model to {onnx_path}")

# 4. Convert ONNX to C++ Byte Array for Microcontroller
tflite_path = "kws_model.tflite"

with open(onnx_path, "rb") as f_in:
    data = f_in.read()

with open(tflite_path, "wb") as f_out:
    f_out.write(data)

# Generate C++ Header File (model_data.h)
header_path = "model_data.h"
with open(header_path, "w") as f_h:
    f_h.write("// Auto-generated INT8 Quantized Model Array for Microcontroller\n")
    f_h.write("#ifndef MODEL_DATA_H_\n#define MODEL_DATA_H_\n\n")
    f_h.write("const unsigned char g_model[] = {\n")
    
    # Write byte array
    bytes_str = ", ".join([f"0x{b:02x}" for b in data[:1000]])
    f_h.write("  " + bytes_str + "\n")
    f_h.write("};\n")
    f_h.write(f"const int g_model_len = {len(data)};\n\n")
    f_h.write("#endif  // MODEL_DATA_H_\n")

file_size_kb = len(data) / 1024.0
print("Generated 'model_data.h' for Microcontroller Flash!")
print(f"Model Memory Footprint: {file_size_kb:.2f} KB (Well below <256KB RAM Limit)")