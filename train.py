import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
from model import get_model
from data_loader import get_data_loaders


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Measures overlap between predicted landslide probability and ground truth masks.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss function: Dice Loss + Binary Cross Entropy.
    This combination works well for landslide segmentation tasks.
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce


def calculate_iou(predictions, targets):
    """
    Calculate Intersection over Union (IoU) metric.
    """
    predictions = (predictions > 0.5).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    if union == 0:
        return 1.0
    
    iou = intersection / union
    return iou.item()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        running_loss += loss.item()
        batch_iou = calculate_iou(outputs, masks)
        running_iou += batch_iou
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item(), 'iou': batch_iou})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_iou = running_iou / len(train_loader)
    
    return epoch_loss, epoch_iou


def validate(model, val_loader, criterion, device):
    """
    Validate the model on validation set.
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            batch_iou = calculate_iou(outputs, masks)
            running_iou += batch_iou
            
            pbar.set_postfix({'loss': loss.item(), 'iou': batch_iou})
    
    epoch_loss = running_loss / len(val_loader)
    epoch_iou = running_iou / len(val_loader)
    
    return epoch_loss, epoch_iou


def train_model(
    num_epochs=50,
    batch_size=16,
    learning_rate=0.001,
    image_size=(256, 256),
    save_dir='checkpoints'
):
    """
    Main training function.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Get data loaders
    print('Loading datasets...')
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0
    )
    
    # Create model
    print('Creating model...')
    model = get_model(device)
    
    # Loss function and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }
    
    best_val_iou = 0.0
    
    print('Starting training...')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        # Print epoch summary
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'history': history
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'New best model saved! Val IoU: {val_iou:.4f}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'history': history
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print('\nTraining completed!')
    print(f'Best validation IoU: {best_val_iou:.4f}')
    
    return model, history


if __name__ == '__main__':
    # Training parameters (optimized for fast training)
    NUM_EPOCHS = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    IMAGE_SIZE = (128, 128)
    
    # Start training
    model, history = train_model(
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        image_size=IMAGE_SIZE
    )

