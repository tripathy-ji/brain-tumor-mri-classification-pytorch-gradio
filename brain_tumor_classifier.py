"""
Brain Tumor MRI Classifier using PyTorch
Author: Your Name
Date: 2025-08-08

- Classifies brain MRI images as tumor/healthy using CNN or pretrained model
- Modular, robust, and ready for deployment/experimentation
"""
import os
import sys
import zipfile
import shutil
import random
import argparse
import csv
import logging
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# Set up logging
logging.basicConfig(filename='train.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Constants
DATASET_ZIP = 'brain_tumor_dataset.zip'
EXTRACTED_DIR = 'brain_tumor_dataset'
IMAGE_SIZE = 224
RANDOM_SEED = 42


def unzip_dataset(zip_path=DATASET_ZIP, extract_to=EXTRACTED_DIR):
    """Extracts the dataset zip file if not already extracted."""
    if not os.path.exists(zip_path):
        print(Fore.RED + f"[ERROR] Dataset zip file '{zip_path}' not found.")
        sys.exit(1)
    if os.path.exists(extract_to) and os.path.isdir(extract_to):
        print(Fore.YELLOW + f"[INFO] Dataset already extracted at '{extract_to}'.")
        return
    print(Fore.GREEN + f"[INFO] Extracting '{zip_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(Fore.GREEN + f"[INFO] Extraction complete.")


def get_transforms():
    """Returns train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def load_datasets(data_dir=EXTRACTED_DIR, val_split=0.2, batch_size=32):
    """Loads datasets and returns dataloaders and class names."""
    train_transform, val_transform = get_transforms()
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    class_names = full_dataset.classes
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
    # Set validation transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, class_names


def build_model(model_name='resnet18', pretrained=True):
    """Builds and returns the model."""
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1)
        )
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 1)
    else:
        raise ValueError('Unsupported model name')
    return model


def train_model(model, train_loader, val_loader, device, epochs=15, lr=1e-4, patience=5):
    """Trains the model with early stopping."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0
    best_model_wts = None
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.sigmoid(outputs) > 0.5
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.byte())
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs) > 0.5
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.byte())
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        # Logging
        print(Fore.CYAN + f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f}")
        logging.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_epoch_loss:.4f}, "
                     f"Train Acc={epoch_acc:.4f}, Val Acc={val_epoch_acc:.4f}")
        # Early stopping
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = model.state_dict()
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(Fore.YELLOW + f"[INFO] Early stopping at epoch {epoch+1}")
                break
    # Save metrics
    with open('metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        for i in range(len(train_losses)):
            writer.writerow([i+1, train_losses[i], val_losses[i], train_accs[i], val_accs[i]])
    # Restore best weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    return model, train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, loader, device):
    """Evaluates the model and returns y_true, y_pred, y_prob."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def predict_image(model, image_path, device, class_names):
    """Predicts a single image and returns label and confidence."""
    from PIL import Image
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        pred = int(prob > 0.5)
    return class_names[pred], prob


def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """Plots loss and accuracy curves."""
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
    plt.savefig('metrics.png')
    plt.show()


def show_sample_predictions(model, loader, device, class_names, n=6):
    """Displays sample predictions with images, labels, and confidence."""
    import matplotlib.pyplot as plt
    model.eval()
    images_shown = 0
    plt.figure(figsize=(15, 6))
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            for i in range(inputs.size(0)):
                if images_shown >= n:
                    break
                img = inputs[i].cpu().permute(1,2,0).numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                plt.subplot(2, n//2, images_shown+1)
                plt.imshow(img)
                plt.title(f"Pred: {class_names[preds[i]]}\nConf: {probs[i]:.2f}")
                plt.axis('off')
                images_shown += 1
            if images_shown >= n:
                break
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Brain Tumor MRI Classifier')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'mobilenet_v2'], help='Model architecture')
    args = parser.parse_args()

    print(Fore.BLUE + f"[INFO] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unzip_dataset()
    data_root = os.path.join(EXTRACTED_DIR)
    if not os.path.exists(data_root):
        print(Fore.RED + f"[ERROR] Extracted data folder '{data_root}' not found.")
        sys.exit(1)
    train_loader, val_loader, class_names = load_datasets(data_root, batch_size=args.batch_size)
    print(Fore.GREEN + f"[INFO] Classes: {class_names}")
    model = build_model(args.model)
    model = model.to(device)
    print(Fore.GREEN + f"[INFO] Model built: {args.model}")
    summary(model, (3, IMAGE_SIZE, IMAGE_SIZE))
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr)
    print(Fore.GREEN + "[INFO] Training complete. Best model saved as 'best_model.pth'.")
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    # Evaluation
    y_true, y_pred, y_prob = evaluate_model(model, val_loader, device)
    print(Fore.YELLOW + "\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(Fore.YELLOW + "\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    # Show sample predictions
    show_sample_predictions(model, val_loader, device, class_names)

if __name__ == '__main__':
    main()
