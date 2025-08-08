# export.py
"""
Model export utilities for TorchScript and ONNX.
"""
import torch
from config import device, IMG_SIZE, SCRIPT_PATH, ONNX_PATH

def export_model(model, script_path=SCRIPT_PATH, onnx_path=ONNX_PATH):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    # TorchScript
    traced = torch.jit.trace(model, dummy)
    traced.save(script_path)
    # ONNX
    try:
        import onnx
        torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["output"],
                          dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                          opset_version=11)
        print(f"Model exported to TorchScript ({script_path}) and ONNX ({onnx_path}).")
    except ImportError:
        print("ONNX export skipped: onnx package not installed.")
