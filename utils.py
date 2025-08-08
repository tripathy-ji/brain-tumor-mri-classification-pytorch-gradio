# utils.py
"""
Utility functions for reproducibility, config saving, and environment logging.
"""
import json
import os
import platform
import torch
import random
import numpy as np
from datetime import datetime

def save_config(config_dict, path="config.json"):
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=4)

def log_environment(log_path="env_log.txt"):
    info = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "datetime": datetime.now().isoformat(),
        "random_seed": random.getstate(),
        "numpy_seed": np.random.get_state()[1][0],
        "torch_seed": torch.initial_seed(),
    }
    with open(log_path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
