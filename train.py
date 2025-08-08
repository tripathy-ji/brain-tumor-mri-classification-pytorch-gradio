# train.py
"""
Training loop with early stopping (val_loss & val_acc), gradient clipping, mixed-precision, OneCycleLR, and metric monitoring.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score, BinaryAUROC, MulticlassAUROC
from sklearn.metrics import confusion_matrix
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
import os
from config import device, EPOCHS, PATIENCE, LR, MODEL_PATH, SEED

class EarlyStopping:
    def __init__(self, patience=5, mode='min', delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif (self.mode == 'min' and score < self.best_score - self.delta) or (self.mode == 'max' and score > self.best_score + self.delta):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Main training loop

def train(model, train_loader, val_loader, num_classes, epochs=EPOCHS, patience=PATIENCE, lr=LR, model_path=MODEL_PATH, grad_clip=1.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    scaler = GradScaler()
    if num_classes == 2:
        f1_metric = BinaryF1Score().to(device)
        auc_metric = BinaryAUROC().to(device)
    else:
        f1_metric = MulticlassF1Score(num_classes=num_classes).to(device)
        auc_metric = MulticlassAUROC(num_classes=num_classes).to(device)
    early_stop_loss = EarlyStopping(patience=patience, mode='min')
    early_stop_acc = EarlyStopping(patience=patience, mode='max')
    train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs = [], [], [], [], [], [], [], []
    best_model_wts = None
    best_val_acc = -float('inf')
    for epoch in range(epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        running_f1, running_auc = 0.0, 0.0
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
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                running_f1 += f1_metric(preds, labels).item() * inputs.size(0)
                try:
                    running_auc += auc_metric(preds, labels).item() * inputs.size(0)
                except Exception:
                    pass
                progress.update(task, advance=1, loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        epoch_f1 = running_f1 / len(train_loader.dataset)
        epoch_auc = running_auc / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        train_f1s.append(epoch_f1)
        train_aucs.append(epoch_auc)
        # Validation
        model.eval()
        val_loss, val_corrects = 0.0, 0
        val_f1, val_auc = 0.0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels)
                val_f1 += f1_metric(preds, labels).item() * inputs.size(0)
                try:
                    val_auc += auc_metric(preds, labels).item() * inputs.size(0)
                except Exception:
                    pass
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        val_epoch_f1 = val_f1 / len(val_loader.dataset)
        val_epoch_auc = val_auc / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        val_f1s.append(val_epoch_f1)
        val_aucs.append(val_epoch_auc)
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, Val Acc={val_epoch_acc:.4f}, Train F1={epoch_f1:.4f}, Val F1={val_epoch_f1:.4f}, Train AUC={epoch_auc:.4f}, Val AUC={val_epoch_auc:.4f}")
        # Early stopping
        early_stop_loss(val_epoch_loss, epoch)
        early_stop_acc(val_epoch_acc, epoch)
        if (early_stop_loss.early_stop or early_stop_acc.early_stop):
            print(f"Early stopping at epoch {epoch+1}")
            break
        # Save best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, model_path)
    # Final best model
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    else:
        # If no improvement, save current weights
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, model_path)
    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, train_aucs, val_aucs
