# Landslide Segmentation with U-Net

This project implements a U-Net deep learning model for landslide segmentation. The model predicts landslide probability maps (0-1 range) for each pixel in satellite/aerial images, indicating the likelihood of landslide occurrence at that location.

## Dataset Structure

The dataset should be organized as follows:
```
dataset/
├── train/
│   ├── images/
│   └── masks/
├── validation/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model, run:
```bash
python train.py
```

You can modify the training parameters in `train.py`:
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `LEARNING_RATE`: Learning rate for optimizer (default: 0.001)
- `IMAGE_SIZE`: Input image size (default: (256, 256))

The trained model will be saved in the `checkpoints/` directory.

### Evaluation

To evaluate the trained model on the test set:
```bash
python evaluate.py
```

This will:
- Load the best model from `checkpoints/best_model.pth`
- Evaluate on the test set and print metrics (IoU, Dice Coefficient, Pixel Accuracy)
- Generate visualization images in the `predictions/` directory

## Model Output

The model outputs **landslide probability maps** where:
- Each pixel value (0-1) represents the probability of that pixel being part of a landslide
- Values closer to 1 indicate higher landslide probability
- Values closer to 0 indicate lower landslide probability
- Binary masks can be created by thresholding at 0.5

## Model Architecture

The U-Net architecture consists of:
- **Encoder (Downsampling)**: Extracts features from input images using convolutional blocks
- **Decoder (Upsampling)**: Reconstructs landslide probability maps using transposed convolutions
- **Skip Connections**: Preserves spatial information by connecting encoder and decoder layers

## Loss Function

The model uses a combined loss function:
- **Dice Loss**: Measures overlap between predicted landslide probability and ground truth masks
- **Binary Cross Entropy**: Standard pixel-wise classification loss
- Combined weight: 50% Dice + 50% BCE

## Metrics

The model is evaluated using:
- **IoU (Intersection over Union)**: Measures overlap between predictions and ground truth
- **Dice Coefficient**: Similar to IoU, measures similarity
- **Pixel Accuracy**: Percentage of correctly classified pixels

## Single Image Prediction

To predict landslide probability for a single image:
```bash
python predict.py --image path/to/image.png --checkpoint checkpoints/best_model.pth
```

This will:
- Generate a landslide probability map (saved as PNG)
- Create a visualization showing input image, probability map, and binary mask
- Print statistics (max probability, mean probability, landslide area percentage)

## Files

- `data_loader.py`: Dataset class and data loading utilities for images and masks
- `model.py`: U-Net model architecture for landslide probability prediction
- `train.py`: Training script with loss functions and training loop
- `evaluate.py`: Evaluation script that generates landslide probability maps and visualizations
- `predict.py`: Single image prediction script for landslide probability
- `requirements.txt`: Python dependencies

