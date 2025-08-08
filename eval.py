# eval.py
"""
Evaluation, confusion matrix, F1, ROC-AUC, and misclassified image logging.
"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from config import device, MISCLASSIFIED_DIR
from PIL import Image

def evaluate(model, loader, classes, save_misclassified=True):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    misclassified = []
    os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, 1)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy()[:, 1] if probs.shape[1] > 1 else probs.cpu().numpy()[:, 0])
            if save_misclassified:
                for j in range(inputs.size(0)):
                    if preds[j] != labels[j]:
                        img = inputs[j].cpu().numpy().transpose(1,2,0)
                        img = (img * 0.229 + 0.485).clip(0,1) * 255
                        img = Image.fromarray(img.astype(np.uint8))
                        img.save(os.path.join(MISCLASSIFIED_DIR, f"misclassified_{i}_{j}.png"))
                        misclassified.append((i, j))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    try:
        auc = roc_auc_score(y_true, y_probs)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass
    return y_true, y_pred, y_probs, misclassified
