"""
Brain Tumor MRI Classifier (PyTorch, Clean, Modular, GPU-ready)
Author: Your Name
Date: 2025-08-08

- High-performance CNN for brain tumor MRI classification
- Stratified split, data augmentation, early stopping, checkpointing
- Exports model (TorchScript/ONNX), single image prediction
- Progress bars, reproducibility, clean code
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

# 1. Set seeds and device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Paths and config
DATA_DIR = r"E:/projects/Brain Tumor MRI Classification & PINN-Ready Preprocessing with Kaggle Dataset (Live Tracking)/archive"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 20
PATIENCE = 5
LR = 1e-4
MODEL_PATH = "best_model.pth"
ONNX_PATH = "brain_tumor_classifier.onnx"

# 3. Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Stratified split helper
def stratified_split(dataset, val_ratio=0.2):
    targets = [s[1] for s in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

# 5. Data loaders
def get_loaders():
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    train_set, val_set = stratified_split(full_dataset, val_ratio=0.2)
    # Set val transform
    val_set.dataset.transform = val_transform
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, full_dataset.classes

# 6. CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE//8) * (IMG_SIZE//8), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 7. Training and validation

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, model_path):
    best_acc = 0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        with Progress(
            TextColumn("[bold blue]Epoch {}/{}".format(epoch+1, epochs)),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("Loss: {task.fields[loss]:.4f}"),
            refresh_per_second=5
        ) as progress:
            task = progress.add_task("train", total=len(train_loader), loss=0.0)
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                progress.update(task, advance=1, loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        scheduler.step(val_epoch_loss)
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, Val Acc={val_epoch_acc:.4f}")
        # Early stopping
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return train_losses, val_losses, train_accs, val_accs

# 8. Evaluation

def evaluate(model, loader, classes):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return y_true, y_pred

# 9. Plotting

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
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

# 10. Export model

def export_model(model, model_path=MODEL_PATH, onnx_path=ONNX_PATH):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    # TorchScript
    traced = torch.jit.trace(model, dummy)
    traced.save("brain_tumor_classifier_script.pt")
    # ONNX
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)
    print(f"Model exported to TorchScript and ONNX.")

# 11. Single image prediction

def predict_image(model, image_path, classes):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = val_transform
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        prob = torch.softmax(output, 1)[0, pred].item()
    print(f"Predicted: {classes[pred]} (confidence: {prob:.2f})")
    plt.imshow(img)
    plt.title(f"Pred: {classes[pred]}\nConf: {prob:.2f}")
    plt.axis('off')
    plt.show()
    return classes[pred], prob

# 12. Main

def main():
    train_loader, val_loader, classes = get_loaders()
    model = SimpleCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    train_losses, val_losses, train_accs, val_accs = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, PATIENCE, MODEL_PATH)
    print(f"Best model saved to {MODEL_PATH}")
    # Load best model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    y_true, y_pred = evaluate(model, val_loader, classes)
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    export_model(model)
    # Predict on a single image (example)
    # predict_image(model, "path_to_image.jpg", classes)

if __name__ == "__main__":
    main()
