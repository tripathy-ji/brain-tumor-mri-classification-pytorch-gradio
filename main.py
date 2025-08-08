# main.py
"""
Main entry point for Brain Tumor MRI Classifier training and evaluation.
"""
from config import set_seed, SEED, MODEL_PATH, METRICS_PATH
from data import get_loaders
from model import SimpleCNN
from train import train
from eval import evaluate
from export import export_model
from utils import save_config, log_environment
import matplotlib.pyplot as plt
import torch

def plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.tight_layout()
    plt.savefig(METRICS_PATH)
    plt.show()

def main():
    set_seed(SEED)
    log_environment()
    train_loader, val_loader, classes = get_loaders()
    model = SimpleCNN(num_classes=len(classes)).to('cuda' if torch.cuda.is_available() else 'cpu')
    config_dict = {
        'num_classes': len(classes),
        'classes': classes,
        'seed': SEED
    }
    save_config(config_dict)
    results = train(model, train_loader, val_loader, num_classes=len(classes))
    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs = results
    print(f"Best model saved to {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    y_true, y_pred, y_probs, misclassified = evaluate(model, val_loader, classes)
    plot_metrics(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s)
    export_model(model)
    # For single image prediction, use predict.py

if __name__ == "__main__":
    main()
