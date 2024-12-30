# Neural-Network-Image-Classification-with-PyTorch

A PyTorch-based image classification project featuring multiple neural network implementations, from simple feedforward networks to convolutional neural networks (CNNs). The project includes both basic and advanced architectures for comparison.

## Project Overview

This project implements and compares different neural network architectures for image classification:
1. Part 1: Simple feedforward neural network
2. Part 2: Advanced CNN architecture with modern techniques

## Features

### Part 1: Basic Neural Network
- Simple feedforward architecture
- Single hidden layer (128 units)
- ReLU activation
- Adam optimizer
- Input standardization
- Cross-entropy loss

### Part 2: Advanced CNN
- Convolutional neural network architecture
- Features:
  - Multiple convolutional layers (16 and 32 channels)
  - Batch normalization
  - MaxPooling
  - Dropout (0.25)
  - Learning rate scheduling
  - Best model checkpointing
- Advanced data preprocessing
- Improved training methodology

## Technical Details

### Network Architectures

#### Basic Network (Part 1)
```
Input -> Linear(in_size, 128) -> ReLU -> Linear(128, out_size)
```

#### CNN Architecture (Part 2)
```
Conv2d(3, 16) -> BatchNorm -> ReLU -> MaxPool ->
Conv2d(16, 32) -> BatchNorm -> ReLU -> MaxPool ->
Linear(flattened_size, 128) -> ReLU -> Dropout(0.25) -> Linear(128, out_size)
```

### Key Components
- `mp10.py`: Main application file
- `neuralnet_part1.py`: Basic neural network implementation
- `neuralnet_part2.py`: Advanced CNN implementation
- `neuralnet_leaderboard.py`: Leaderboard version template
- `utils.py`: Utility functions for data handling and metrics
- `reader.py`: Data loading and preprocessing functions

## Usage

Run the training script with different configurations:

```bash
python mp10.py [options]
```

### Command Line Arguments
- `--dataset`: Path to the dataset file
- `--part1`: Run the basic neural network
- `--part2`: Run the advanced CNN
- `--leaderboard`: Run the leaderboard version
- `--epochs`: Number of training epochs (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)

## Data Format

The project expects data in pickle format with the following structure:
- `data`: Image data in numpy array format
- `labels`: Corresponding class labels

## Performance Metrics

The system outputs:
- Overall accuracy
- Confusion matrix
- Number of model parameters
- Training loss over epochs

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Pickle

## Installation

```bash
pip install torch numpy
```

## Implementation Details

### Data Preprocessing
- Standardization using mean and standard deviation
- Reshaping for CNN input (3x31x31 for images)
- Batch processing with DataLoader

### Training Features
- Batch training
- Learning rate scheduling
- Model checkpointing
- Early stopping based on validation loss
- Data standardization

## Acknowledgments

This project was developed as part of the CS440/ECE448 course at the University of Illinois at Urbana-Champaign.
