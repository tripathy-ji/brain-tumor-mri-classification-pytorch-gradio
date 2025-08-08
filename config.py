# config.py
"""
Configuration and reproducibility settings for Brain Tumor MRI Classifier
"""
import os
import random
import numpy as np
import torch

# Set all seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths and hyperparameters
DATA_DIR = r"E:/projects/Brain Tumor MRI Classification & PINN-Ready Preprocessing with Kaggle Dataset (Live Tracking)/archive"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 20
PATIENCE = 5
LR = 1e-4
MODEL_PATH = "best_model.pth"
ONNX_PATH = "brain_tumor_classifier.onnx"
SCRIPT_PATH = "brain_tumor_classifier_script.pt"
METRICS_PATH = "metrics.png"
MISCLASSIFIED_DIR = "misclassified"
SEED = 42
CONFIDENCE_THRESHOLD = 0.5
NUM_WORKERS = 2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
