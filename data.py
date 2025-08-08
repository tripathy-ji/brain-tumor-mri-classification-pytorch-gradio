# data.py
"""
Data loading, augmentation (Albumentations), and stratified split for Brain Tumor MRI Classifier
"""
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from albumentations import Compose, Resize, RandomRotate90, HorizontalFlip, VerticalFlip, Transpose, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast, Normalize
from albumentations.pytorch import ToTensorV2
from PIL import Image
from config import DATA_DIR, BATCH_SIZE, IMG_SIZE, NUM_WORKERS, SEED

class AlbumentationsTransform:
    def __init__(self, augment=True):
        if augment:
            self.transform = Compose([
                Resize(IMG_SIZE, IMG_SIZE),
                RandomRotate90(),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Transpose(p=0.5),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
                HueSaturationValue(p=0.3),
                RandomBrightnessContrast(p=0.3),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = Compose([
                Resize(IMG_SIZE, IMG_SIZE),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    def __call__(self, img):
        img = np.array(img)
        return self.transform(image=img)["image"]

def stratified_split(dataset, val_ratio=0.2):
    targets = [s[1] for s in dataset.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def get_loaders():
    train_transform = AlbumentationsTransform(augment=True)
    val_transform = AlbumentationsTransform(augment=False)
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    train_set, val_set = stratified_split(full_dataset, val_ratio=0.2)
    # Set val transform for validation set
    val_set.dataset.transform = val_transform
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, full_dataset.classes
