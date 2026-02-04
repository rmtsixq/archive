import torch
import torch.nn as nn
from model import get_model
import os

def export_model():
    # Load model
    device = torch.device('cpu')
    checkpoint_path = 'checkpoints/best_model.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found")
        return

    model = get_model(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create dummy input
    dummy_input = torch.randn(1, 3, 128, 128, device=device)

    # Export to ONNX
    output_path = 'model.onnx'
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path, 
            verbose=True,
            input_names=['input'], 
            output_names=['output'],
            opset_version=11
        )
        print(f"Model successfully exported to {output_path}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")

if __name__ == '__main__':
    export_model()
