# predict.py
"""
Batch and single image prediction with confidence threshold.
"""
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from config import device, CONFIDENCE_THRESHOLD
from data import AlbumentationsTransform

def predict_image(model, image_path, classes, threshold=CONFIDENCE_THRESHOLD):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = AlbumentationsTransform(augment=False)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, 1)
        conf, pred = torch.max(probs, 1)
        conf = conf.item()
        pred = pred.item()
    print(f"Predicted: {classes[pred]} (confidence: {conf:.2f})")
    plt.imshow(img)
    plt.title(f"Pred: {classes[pred]}\nConf: {conf:.2f}")
    plt.axis('off')
    plt.show()
    return classes[pred], conf, conf >= threshold

def predict_batch(model, image_paths, classes, threshold=CONFIDENCE_THRESHOLD):
    results = []
    for path in image_paths:
        label, conf, passed = predict_image(model, path, classes, threshold)
        results.append((path, label, conf, passed))
    return results
