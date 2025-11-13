# Image Classification with Various Machine Learning Models

## Overview

This repository implements and compares several machine learning models for image classification using the CIFAR-10 dataset. The models include:

- **Gaussian Naive Bayes**
- **Decision Tree**
- **Multi-Layer Perceptron (MLP)**
- **VGG11 Convolutional Neural Network (CNN)**

The goal is to evaluate the performance of these models on the same dataset and analyze the results.

## Project Structure

```
📂 image-classification/
├── 📂 data/
│   ├── 📂 features/ - Extracted features
│   ├── 📂 raw/ - CIFAR-10 images
|   └── 📂 subsets/ - CIFAR-10 subsets
|
├── 📂 models/
│   ├── 📂 gnb/ - Gaussian Naive Bayes models + implementation
│   ├── 📂 decision_tree/ - Decision tree models + implementation
│   ├── 📂 mlp/ - Multi-Layer Perceptron models + implemetnation
│   └── 📂 cnn/ - VGG11 Convolutional Neural Network model + implementation
|
├── 📂 report/ - Final written report
├── 📂 results/
│   ├── 📂 gnb/ - Results for Gaussian Naive Bayes models
│   ├── 📂 decision_tree/ - Results for Decision Tree models
│   ├── 📂 mlp/ - Results for Multi-Layer Perceptron models
│   └── 📂 cnn/ - Results VGG11 Convolutional Neural Network model
|
├── 📂 tools/ - Utility scripts
└── 📄 preprocess.py - Script for loading, resizing, normalizing and extracting features
```

## Datasets

| Dataset Type      | Location         | Samples | Description                                                                                                                    | Files                                                |
| ----------------- | ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Raw CIFAR-10 Data | `data/raw/`      | 60,000  | The CIFAR-10 dataset containing 50,000 training and 10,000 test RGB images belonging to 10 object classes of size 32 × 32 × 3. | `cifar-10-batches-py/`                               |
| CIFAR-10 Subsets  | `data/subsets/`  | 6,000   | Subset of CIFAR-10 with 6000 images (500 train, 100 test per class) for quicker experimentation.                               | `cifar10_test_100.pt`<br> `cifar10_train_500.pt`     |
| Feature Vectors   | `data/features/` | 6,000   | Uses a pre-trained ResNet-18 CNN to extract 512 × 1 feature vectors for the RGB images.                                        | `resnet18_test_512.npz`<br> `resnet18_train_512.npz` |
| PCA Features      | `data/features/` | 6,000   | Uses PCA in scikit learn to further reduce the size of feature vectors from 512×1 to 50×1.                                     | `pca_test_50.npz`<br> `pca_train_50.npz`             |

## Models

| Algorithm              | Dataset Used     | Features             | Location                                                                                             |
| ---------------------- | ---------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| Gaussian Naive Bayes   | CIFAR-10 Subsets | PCA Features (50-D)  | `models/gnb/custom_gnb.pkl`<br> `models/gnb/sklearn_gnb.pkl`                                         |
| Decision Tree          | CIFAR-10 Subsets | PCA Features (50-D)  | `models/decision_tree/custom_decision_tree.pkl`<br> `models/decision_tree/sklearn_decision_tree.pkl` |
| Multi-Layer Perceptron | CIFAR-10 Subsets | PCA Features (50-D)  | `models/mlp/custom_mlp.pt`<br> `models/mlp/sklearn_mlp.pkl`                                          |
| VGG11 CNN              | CIFAR-10 Subsets | Raw Images (32x32x3) | `models/cnn/vgg11_custom.pt`                                                                         |

## Setup Instructions

### Prerequisites

- Python 3.7+
- Pytorch
- scikit-learn
- NumPy

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/matteorobidoux/image-classification.git
   cd image-classification
   ```
2. **Download and prepare the CIFAR-10 dataset:**
   ```bash
   python preprocess.py
   ```
3. **Train the models**
   ```bash
   python -m models.gnb.gnb_train
   python -m models.decision_tree.decision_tree_train
   python -m models.mlp.mlp_train
   python -m models.cnn.cnn_train
   ```
4. **Evaluate the models**
   ```bash
   python -m models.gnb.gnb_evaluate
   python -m models.decision_tree.decision_tree_evaluate
   python -m models.mlp.mlp_evaluate
   python -m models.cnn.cnn_evaluate
   ```

## Results

Performance metrics, confusion matrices, and visualizations for each model can be found in:

```
results/
├── gnb/
├── decision_tree/
├── mlp/
└── cnn/
```

## Report

The full analysis of the models and findings can be found in:

```
report/final_report.pdf
```

## Contributors

- **Matteo Robidoux**
