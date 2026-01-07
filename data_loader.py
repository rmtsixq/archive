import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


class SegmentationDataset(Dataset):
    """
    Custom dataset class for image segmentation.
    Loads images and their corresponding masks from separate directories.
    """
    def __init__(self, images_dir, masks_dir, transform=None, image_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_size = image_size
        
        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Get corresponding mask name (replace 'image_' with 'mask_')
        mask_name = img_name.replace('image_', 'mask_')
        mask_path = os.path.join(self.masks_dir, mask_name)
        
        # Open and convert to RGB
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        
        # Resize to fixed size
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = mask.resize(self.image_size, Image.NEAREST)
        
        # Convert to numpy arrays
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32) / 255.0
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Binarize mask (threshold at 0.5)
        mask = (mask > 0.5).float()
        
        return image, mask


def get_data_loaders(batch_size=16, image_size=(256, 256), num_workers=2):
    """
    Create data loaders for train, validation, and test sets.
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])
    
    # Create datasets
    train_dataset = SegmentationDataset(
        images_dir='dataset/train/images',
        masks_dir='dataset/train/masks',
        transform=train_transform,
        image_size=image_size
    )
    
    val_dataset = SegmentationDataset(
        images_dir='dataset/validation/images',
        masks_dir='dataset/validation/masks',
        transform=None,
        image_size=image_size
    )
    
    test_dataset = SegmentationDataset(
        images_dir='dataset/test/images',
        masks_dir='dataset/test/masks',
        transform=None,
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

