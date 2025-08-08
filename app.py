"""
Brain Tumor MRI Classifier Web App (Gradio)
- Loads trained PyTorch model (best_model.pth)
- Accepts user-uploaded MRI images
- Preprocesses images as during training
- Predicts 'Tumor' or 'No Tumor' with confidence
- Clean Gradio UI, GPU/CPU support, error handling, server logs
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
import gradio as gr
import numpy as np
import logging

# --- CONFIGURATION ---
MODEL_PATH = "best_model.pth"
IMG_SIZE = 224
CLASS_NAMES = ["No Tumor", "Tumor"]  # 0: no, 1: yes
# If you used custom normalization, update these values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# --- MODEL DEFINITION (must match training) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- LOAD MODEL ---
def load_model(model_path=MODEL_PATH):
    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    logging.info("Model loaded successfully.")
    return model

model = load_model()

# --- PREPROCESSING ---
def preprocess_image(img: Image.Image):
    if img.mode != "RGB":
        img = img.convert("RGB")
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)

# --- PREDICTION ---
def predict(img):
    try:
        if img is None:
            raise ValueError("No image uploaded.")
        tensor = preprocess_image(img)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(probs[pred_idx])
        logging.info(f"Prediction: {pred_label} (Confidence: {confidence:.4f})")
        return {
            "Predicted Label": pred_label,
            "Confidence": f"{confidence*100:.2f}%"
        }
    except UnidentifiedImageError:
        logging.error("Uploaded file is not a valid image.")
        return {"Predicted Label": "Error: Invalid image file.", "Confidence": "N/A"}
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"Predicted Label": f"Error: {str(e)}", "Confidence": "N/A"}

# --- GRADIO INTERFACE ---
description = """
# 🧠 Brain Tumor MRI Classifier
Upload a brain MRI image (JPG/PNG). The model will predict if a tumor is present.
"""

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="Brain Tumor MRI Classification",
    description=description,
    allow_flagging="never",
    examples=None
)

def main():
    iface.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

if __name__ == "__main__":
    main()
