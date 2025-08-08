import os
from PIL import Image
import logging

def clean_images(root_dir):
    removed = 0
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg')):
                path = os.path.join(subdir, file)
                try:
                    with Image.open(path) as img:
                        img.verify()
                except Exception:
                    os.remove(path)
                    removed += 1
                    logging.info(f"Removed corrupted: {path}")
    print(f"Total corrupted images removed: {removed}")

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        clean_images(sys.argv[1])
    else:
        try:
            from config import DATA_DIR
            clean_images(DATA_DIR)
        except ImportError:
            print("DATA_DIR not found. Please specify the dataset directory as an argument.")
