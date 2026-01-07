import torch
import numpy as np
from PIL import Image
import os
from model import get_model
from torchvision import transforms
import matplotlib.pyplot as plt


def predict_landslide_probability(model, image_path, device, image_size=(256, 256), save_probability_map=True):
    """
    Predict landslide probability for a single image.
    
    Args:
        model: Trained U-Net model
        image_path: Path to input image
        device: Device to run inference on
        image_size: Size to resize image to
        save_probability_map: Whether to save the probability map as PNG
    
    Returns:
        probability_map: 2D numpy array with landslide probabilities (0-1)
        binary_mask: Binary mask (0 or 1) with threshold at 0.5
    """
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Resize image
    image_resized = image.resize(image_size, Image.BILINEAR)
    
    # Convert to tensor
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probability_map = output[0].cpu().squeeze().numpy()  # Shape: (H, W)
    
    # Resize probability map back to original image size
    prob_img = Image.fromarray((probability_map * 255).astype(np.uint8), mode='L')
    prob_img_resized = prob_img.resize(original_size, Image.BILINEAR)
    probability_map = np.array(prob_img_resized).astype(np.float32) / 255.0
    
    # Create binary mask
    binary_mask = (probability_map > 0.5).astype(np.float32)
    
    # Save probability map if requested
    if save_probability_map:
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        prob_path = os.path.join(output_dir, f'landslide_prob_{filename}')
        prob_img_resized.save(prob_path)
        print(f'Saved landslide probability map to {prob_path}')
    
    return probability_map, binary_mask


def visualize_prediction(image_path, probability_map, binary_mask, save_path=None):
    """
    Visualize input image with landslide probability map and binary mask.
    """
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_array)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Landslide probability map
    im = axes[1].imshow(probability_map, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Landslide Probability Map')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Probability')
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Binary Landslide Mask (threshold=0.5)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved visualization to {save_path}')
    else:
        plt.show()
    
    plt.close()


def main():
    """
    Main function for single image prediction.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict landslide probability for an image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, 
                       help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f'Error: Checkpoint not found at {args.checkpoint}')
        print('Please train the model first using train.py')
        return
    
    print(f'Loading model from {args.checkpoint}...')
    model = get_model(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully!')
    
    # Predict
    print(f'Predicting landslide probability for {args.image}...')
    probability_map, binary_mask = predict_landslide_probability(
        model, args.image, device, save_probability_map=True
    )
    
    # Calculate statistics
    max_prob = np.max(probability_map)
    mean_prob = np.mean(probability_map)
    landslide_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    landslide_percentage = (landslide_pixels / total_pixels) * 100
    
    print('\n' + '='*50)
    print('Landslide Prediction Results:')
    print('='*50)
    print(f'Maximum probability: {max_prob:.4f}')
    print(f'Mean probability: {mean_prob:.4f}')
    print(f'Landslide area: {landslide_percentage:.2f}%')
    print('='*50)
    
    # Visualize
    if args.output:
        save_path = args.output
    else:
        filename = os.path.basename(args.image)
        save_path = f'predictions/visualization_{filename}'
    
    visualize_prediction(args.image, probability_map, binary_mask, save_path)


if __name__ == '__main__':
    main()

