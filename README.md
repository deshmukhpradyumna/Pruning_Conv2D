# Pruning Techniques on Convolutional Neural Networks
This project explores unstructured and structured pruning methods to reduce computational costs in CNNs while maintaining accuracy. Experiments were conducted on MNIST and CIFAR-10 datasets using custom implementations. Below are key features and implementation details.

# Key Features
Unstructured Pruning
Applied to a 2-layer CNN on MNIST using polynomial decay scheduling, achieving 50% sparsity without significant accuracy drop

Structured Pruning
Implemented on CIFAR-10 CNN using filter norm analysis, reducing parameters from 767,936 to 380,528 (≈50% reduction) while preserving baseline accuracy

# Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/pruning-project.git
$bash cd pruning-project
pip install -r requirements.txt
```
# Usage
1. Unstructured Pruning (MNIST)
Run with default parameters:
```bash
python unstructured_pruning.py
```
Parameters:

--sparsity_target: Target sparsity ratio (default=0.5)

--epochs: Training epochs (default=10)

--batch_size: Batch size (default=128)

# 2. Structured Pruning (CIFAR-10)
Execute with:
```bash
python structured_pruning.py
```
Parameters:

--pruning_threshold: Filter norm cutoff (default=0.5)

--fine_tune_epochs: Fine-tuning epochs post-pruning (default=5)

# Results
Unstructured Pruning Performance
<img src="unstructured_mnist.png" alt="Unstructured Pruning Results" width="400"/> *Achieved 50% sparsity with <1% accuracy drop on MNIST test set*
Structured Pruning Efficiency
<img src="structured_cifar.png" alt="Structured Pruning Results" width="400"/> *Parameter reduction from 767K to 380K while matching baseline accuracy on CIFAR-10*

# Implementation Details
Unstructured Pruning: Uses iterative magnitude-based weight pruning with polynomial decay scheduler

Structured Pruning: Implements filter-level pruning via L2-norm thresholding followed by fine-tuning

Code Structure:
```txt
pruning-project/
├── unstructured_pruning.py    # MNIST weight pruning
├── structured_pruning.py      # CIFAR-10 filter pruning
├── models/                    # CNN architectures
└── results/                   # Output visualizations
```
