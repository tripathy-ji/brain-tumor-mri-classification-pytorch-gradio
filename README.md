
# Brain Tumor MRI Classification & PINN-Ready Preprocessing with Kaggle Dataset (Live Tracking)

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1%2B%20(CUDA%2011.8)-ee4c2c)
![CUDA](https://img.shields.io/badge/cuda-11.8-green)
![License](https://img.shields.io/badge/license-MIT-green)
![Kaggle Dataset](https://img.shields.io/badge/dataset-kaggle-blue)

> **A robust, from-scratch deep learning pipeline for brain tumor MRI classification with live training tracking and PINN-ready preprocessing.**

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Environment Setup](#environment-setup)
- [Requirements](#requirements)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Exporting Model](#exporting-model)
- [Troubleshooting](#troubleshooting)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project implements a deep learning pipeline for classifying brain MRI images as `tumor` or `no tumor` using a custom Convolutional Neural Network (CNN) built entirely from scratch (no pre-trained weights). The goal is to achieve high accuracy (target 90%+) and low loss on the Kaggle brain MRI dataset. The training process features live tracking with Rich progress bars, and the preprocessing pipeline is designed to be PINN-ready for future integration with physics-informed neural networks.

---

## Dataset

- **Source:** [Kaggle - Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **License:** Copyright - dataset authors

**Structure:**
```
archive/
├── Training/
│   ├── yes/   # Tumor images
│   └── no/    # No tumor images
└── Testing/
    ├── yes/
    └── no/
```

**Number of images per class:**
- `yes` (tumor): _[fill in actual count after download]_ 
- `no` (no tumor): _[fill in actual count after download]_

**Preprocessing:**
- All images are resized, normalized, and cleaned (corrupted images removed).
- Advanced augmentation (Albumentations) for robust training.
- Stratified train/val split for balanced evaluation.
- PINN-ready: pipeline can be extended for physics-informed learning.

---

## Features

- Custom CNN architecture (no pre-trained weights)
- Overfitting prevention: dropout, data augmentation, weight decay, early stopping
- CUDA GPU acceleration (GTX 1650 / CUDA 11.8) and TPU (Colab) support
- Live progress tracking with Rich library
- Model export: `.pth` (PyTorch), `.onnx` (optional)
- PINN-ready preprocessing pipeline

---

## Environment Setup

**Supported OS:**
- Windows 10/11
- Linux
- macOS (with GPU)

**Python version:** >=3.9

**Installation:**
```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**GPU/TPU Setup:**
- For CUDA: Install NVIDIA drivers and CUDA 11.8 toolkit
- For Colab TPU: Use `torch_xla` and set device in code

---

## Requirements

- torch
- torchvision
- torchaudio
- rich
- onnx
- scikit-learn
- matplotlib
- pillow
- tqdm

See [`requirements.txt`](./requirements.txt) for full list.

---

## Usage

**1. Download Dataset:**
   - [Kaggle Download Instructions](https://www.kaggle.com/docs/api)
   - Or download manually and extract to `archive/`

**2. Set Dataset Path:**
   In your script:
   ```python
   dataset_path = "E:/projects/Brain Tumor MRI Classification & PINN-Ready Preprocessing with Kaggle Dataset (Live Tracking)/archive"
   ```

**3. Train the Model:**
   ```bash
   python main.py
   ```

**4. Clean Corrupted Images:**
   ```bash
   python clean_images.py
   ```

**5. Predict on a Single Image:**
   ```bash
   python predict.py --image path_to_image.jpg
   ```

---

## Training

- **CNN Architecture:** Custom, with multiple conv layers, batch norm, dropout, and fully connected layers.
- **Hyperparameters:**
  - Batch size: 32 (default)
  - Learning rate: 1e-3 (OneCycleLR)
  - Optimizer: Adam
  - Early stopping: patience=5
  - Weight decay: 1e-4
  - Mixed precision: enabled
- **Device:**
  - GPU: set automatically if available
  - TPU: set manually in code (Colab)

**Rich Progress Example:**
![Rich Progress Example](https://raw.githubusercontent.com/Textualize/rich/main/imgs/progress.gif)

---

## Evaluation

- **Metrics:** Accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix
- **Classification Report:**
  - Printed after training
  - Saved to `metrics.png`
- **Example Output:**
  ```
  accuracy                           0.91      1405
  macro avg       0.90      0.91      0.90      1405
  weighted avg    0.91      0.91      0.91      1405
  ```
- **Confusion Matrix:**
  - Saved and displayed after evaluation

---

## Exporting Model

- **PyTorch (.pth):**
  - Saved automatically after training
- **ONNX (.onnx):**
  - Optional, for deployment/inference acceleration
  - Install ONNX:
    ```bash
    pip install onnx
    ```

---

## Troubleshooting

- **Missing dataset:**
  - Ensure `archive/Training/` and `archive/Testing/` exist
- **ONNX not installed:**
  - Run `pip install onnx`
- **CUDA not found:**
  - Check NVIDIA drivers and CUDA toolkit
- **Other errors:**
  - Check logs and ensure all dependencies are installed

---

## Future Work

- Multi-class tumor type detection
- Integrate PINNs for physics-informed learning
- Deploy with FastAPI or Streamlit

---

## Contributing

1. Fork this repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## License

- **Dataset:** Copyright - dataset authors (see [Kaggle dataset page](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection))
- **Code:** MIT License

---

## Acknowledgements

- [Kaggle dataset authors](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- [PyTorch](https://pytorch.org/)
- [Rich library](https://github.com/Textualize/rich)
