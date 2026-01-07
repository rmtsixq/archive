import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import get_model
from data_loader import get_data_loaders


def calculate_metrics(predictions, targets):
    """
    Calculate various segmentation metrics:
    - IoU (Intersection over Union)
    - Dice Coefficient
    - Pixel Accuracy
    """
    predictions = (predictions > 0.5).float()
    targets = targets.float()
    
    # Intersection and Union
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    # IoU
    if union == 0:
        iou = 1.0
    else:
        iou = (intersection / union).item()
    
    # Dice Coefficient
    if (predictions.sum() + targets.sum()) == 0:
        dice = 1.0
    else:
        dice = (2. * intersection / (predictions.sum() + targets.sum())).item()
    
    # Pixel Accuracy
    correct = (predictions == targets).sum().item()
    total = predictions.numel()
    pixel_acc = correct / total
    
    return {
        'iou': iou,
        'dice': dice,
        'pixel_accuracy': pixel_acc
    }


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test set and return average metrics.
    """
    model.eval()
    all_metrics = {'iou': [], 'dice': [], 'pixel_accuracy': []}
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calculate metrics for each sample in batch
            for i in range(outputs.shape[0]):
                pred = outputs[i:i+1]
                mask = masks[i:i+1]
                metrics = calculate_metrics(pred, mask)
                
                all_metrics['iou'].append(metrics['iou'])
                all_metrics['dice'].append(metrics['dice'])
                all_metrics['pixel_accuracy'].append(metrics['pixel_accuracy'])
    
    # Calculate averages
    avg_metrics = {
        'iou': np.mean(all_metrics['iou']),
        'dice': np.mean(all_metrics['dice']),
        'pixel_accuracy': np.mean(all_metrics['pixel_accuracy'])
    }
    
    return avg_metrics, all_metrics


def save_landslide_probability_map(probability_map, save_path):
    """
    Save landslide probability map as a PNG image.
    Probability values (0-1) are saved as grayscale image (0-255).
    """
    # Convert probability (0-1) to uint8 (0-255)
    prob_uint8 = (probability_map * 255).astype(np.uint8)
    img = Image.fromarray(prob_uint8, mode='L')
    img.save(save_path)
    print(f'Saved landslide probability map to {save_path}')


def visualize_predictions(model, test_loader, device, num_samples=5, save_dir='predictions'):
    """
    Visualize model predictions on test samples.
    Saves both visualization images and landslide probability maps.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'probability_maps'), exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            if idx >= num_samples:
                break
            
            images = images.to(device)
            masks = masks.to(device)
            
            # Get landslide probability predictions
            outputs = model(images)
            
            # Process first image in batch
            image = images[0].cpu().permute(1, 2, 0).numpy()
            mask = masks[0].cpu().squeeze().numpy()
            pred = outputs[0].cpu().squeeze().numpy()  # Landslide probability map (0-1)
            pred_binary = (pred > 0.5).astype(np.float32)
            
            # Save landslide probability map
            prob_map_path = os.path.join(save_dir, 'probability_maps', f'landslide_prob_{idx}.png')
            save_landslide_probability_map(pred, prob_map_path)
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            axes[0].imshow(image)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Ground Truth Landslide Mask')
            axes[1].axis('off')
            
            # Landslide probability map with color scale
            im = axes[2].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title('Landslide Probability Map')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label='Probability')
            
            axes[3].imshow(pred_binary, cmap='gray')
            axes[3].set_title('Binary Landslide Mask (threshold=0.5)')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'prediction_{idx}.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f'Saved {num_samples} prediction visualizations to {save_dir}/')
    print(f'Saved {num_samples} landslide probability maps to {save_dir}/probability_maps/')


def load_model(checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    """
    model = get_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded model from {checkpoint_path}')
    print(f'Model was trained for {checkpoint["epoch"]+1} epochs')
    if 'val_iou' in checkpoint:
        print(f'Best validation IoU: {checkpoint["val_iou"]:.4f}')
    return model


def main():
    """
    Main evaluation function.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        print('Please train the model first using train.py')
        return
    
    model = load_model(checkpoint_path, device)
    
    # Get test loader
    print('Loading test dataset...')
    _, _, test_loader = get_data_loaders(
        batch_size=8,
        image_size=(256, 256),
        num_workers=2
    )
    
    # Evaluate
    print('Evaluating on test set...')
    avg_metrics, all_metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    print('\n' + '='*50)
    print('Test Set Evaluation Results:')
    print('='*50)
    print(f'Mean IoU: {avg_metrics["iou"]:.4f}')
    print(f'Mean Dice Coefficient: {avg_metrics["dice"]:.4f}')
    print(f'Mean Pixel Accuracy: {avg_metrics["pixel_accuracy"]:.4f}')
    print('='*50)
    
    # Visualize predictions
    print('\nGenerating prediction visualizations...')
    visualize_predictions(model, test_loader, device, num_samples=10)


if __name__ == '__main__':
    main()

