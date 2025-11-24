# Comparing Machine Learning Models for CIFAR-10 Classification

## Overview

This repository implements and compares several machine learning models for image classification using the CIFAR-10 dataset. The models include:

- **Gaussian Naive Bayes (GNB)**
- **Decision Tree (DT)**
- **Multi-Layer Perceptron (MLP)**
- **Convolutional Neural Network (CNN)**

The goal is to evaluate and better understand the models on the same dataset and analyze the results.

## Project Structure

```
📂 image-classification/
├── 📂 data/
│   ├── 📂 features/ - Extracted features
│   ├── 📂 raw/ - CIFAR-10 images
|   ├── 📂 subsets/ - CIFAR-10 subsets
|   └── 📂 training/ - Data used for training
|
├── 📂 models/
│   ├── 📂 gnb/
|   |   ├── 🥒 custom_gnb.pkl - Trained custom Gaussian Naive Bayes model
|   |   ├── 🥒sklearn_gnb.pkl - Trained Scikit-learn Gaussian Naive Bayes model
│   │   ├── 📄 train.py - Train Gaussian Naive Bayes models
│   │   ├── 📄 evaluate.py - Evaluate Gaussian Naive Bayes models
│   │   └── 📄 gnb.py - Custom implementation of the Gaussian Naive Bayes model
|
│   ├── 📂 decision_tree/
│   │   ├── 📂 depth_5/ - Trained Decision Tree models with a maximum depth of 5
│   │   ├── 📂 depth_10/ - Trained Decision Tree models with a maximum depth of 10
│   │   ├── 📂 depth_20/ - Trained Decision Tree models with a maximum depth of 20
│   │   ├── 📂 depth_50/ - Trained Decision Tree models with a maximum depth of 50
│   │   ├── 📄 configs.py - Parameters for the Decision Tree models
│   │   ├── 📄 train.py - Train Decision Tree models
│   │   ├── 📄 evaluate.py - Evaluate Decision Tree models
│   │   └── 📄 decision_tree.py - Custom implementation of the Decision Tree model
|
│   ├── 📂 mlp/
│   │   ├── 📂 base/ - Trained MLP models with the base architecture
│   │   ├── 📂 wide/ - Trained MLP models with a wide architecture
│   │   ├── 📂 deep/ - Trained MLP models with a deep architecture
│   │   ├── 📂 shallow/ - Trained MLP models with a shallow architecture
│   │   ├── 📂 single/ - Trained MLP models with a single-layer architecture
│   │   ├── 📄 configs.py - Parameters for the MLP models
│   │   ├── 📄 train.py - Train MLP models
│   │   ├── 📄 evaluate.py - Evaluate MLP models
│   │   └── 📄 mlp.py - Custom implementation of the Multi-Layer Perceptron model
|   |
│   └── 📂 cnn/
│       ├── 📂 shallow/ - Trained CNN models with a shallow architecture
│       ├── 📂 vgg11/ - Trained CNN models with a VGG11 architecture
│       ├── 📂 vgg11_large_kernels/ - Trained CNN models with a large kernel VGG11 architecture
│       ├── 📄 configs.py - Parameters for the CNN models
│       ├── 📄 train.py - Train CNN models
│       ├── 📄 evaluate.py - Evaluate CNN models
│       └── 📄 cnn.py - Custom implementation of the Convolutional Neural Network model
|
├── 📂 report/
│   └── 📄 final_report.pdf - Final report analyzing the models and results
|
├── 📂 results/
│   ├── 📂 gnb/
|   |   ├── 📂 gnb_custom/ - Results for Custom Gaussian Naive Bayes Model
│   |   └── 📂 gnb_sklearn/ - Results for Scikit-learn Gaussian Naive Bayes Model
│   ├── 📂 decision_tree/
│   │   ├── 📂 decision_tree_custom/
|   |   |   ├── 📂 depth_5/ - Results for Custom Decision Tree with a max depth of 5
|   |   |   ├── 📂 depth_10/ - Results for Custom Decision Tree with a max depth of 10
|   |   |   ├── 📂 depth_20/ - Results for Custom Decision Tree with a max depth of 20
|   |   |   └── 📂 depth_50/ - Results for Custom Decision Tree with a max depth of 50
|   |   |
│   │   └── 📂 decision_tree_sklearn/
|   |       ├── 📂 depth_5/ - Results for Sklearn Decision Tree with a max depth of 5
|   |       ├── 📂 depth_10/ - Results for Sklearn Decision Tree with a max depth of 10
|   |       ├── 📂 depth_20/ - Results for Sklearn Decision Tree with a max depth of 20
|   |       └── 📂 depth_50/ - Results for Sklearn Decision Tree with a max depth of 50
|   |
│   ├── 📂 mlp/
│   │   ├── 📂 mlp_custom/ - Results Multi-Layer Perceptron custom mode
|   |   |   ├── 📂 base/ - Results Custom Multi-Layer Perceptron base model
|   |   |   ├── 📂 wide/ - Results Custom Multi-Layer Perceptron wide model
|   |   |   ├── 📂 deep/ - Results Custom Multi-Layer Perceptron deep model
|   |   |   ├── 📂 shallow/ - Results Custom Multi-Layer Perceptron shallow model
|   |   |   └── 📂 single/ - Results Custom Multi-Layer Perceptron single-layer model
|   |   |
│   │   └── 📂 mlp_sklearn/ - Results Multi-Layer Perceptron Scikit-learn model
|   |       ├── 📂 base/ - Results Sklearn Multi-Layer Perceptron base model
|   |       ├── 📂 wide/ - Results Sklearn Multi-Layer Perceptron wide model
|   |       ├── 📂 deep/ - Results Sklearn Multi-Layer Perceptron deep model
|   |       ├── 📂 shallow/ - Results Sklearn Multi-Layer Perceptron shallow model
|   |       └── 📂 single/ - Results Sklearn Multi-Layer Perceptron single-layer model
|   |
│   └── 📂 cnn/ - Results VGG11 Convolutional Neural Network model
|       └── 📂 custom_cnn/
|           ├── 📂 shallow/ - Results Custom CNN shallow model
|           ├── 📂 vgg11/ - Results Custom CNN VGG11 model
|           └── 📂 vgg11_large_kernels/ - Results Custom CNN VGG11 large kernels model
|
├── 📂 tools/
|   ├── 📄 feature_extraction.py - Script for extracting features using pre-trained CNN
|   ├── 📄 pca_utils.py - Script for applying PCA to reduce feature dimensions
|   ├── 📄 cifar10_utils.py - Script for loading the CIFAR-10 data
|   └── 📄 evaluation_utils.py - Script to aid with evaluation
|
└── 📄 preprocess.py - Script for loading, resizing, normalizing and extracting features
```

## Datasets

The following datasets are used and generated in this project:

| Dataset Type           | Location         | Samples | Description                                                                                                                    | Files                                                |
| ---------------------- | ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| Raw CIFAR-10 Data      | `data/raw/`      | 60,000  | The CIFAR-10 dataset containing 50,000 training and 10,000 test RGB images belonging to 10 object classes of size 32 × 32 × 3. | `cifar-10-batches-py/`                               |
| CIFAR-10 Subsets       | `data/subsets/`  | 6,000   | Subset of CIFAR-10 with 6000 images (500 train, 100 test per class) for quicker experimentation.                               | `cifar10_test_100.pt`<br> `cifar10_train_500.pt`     |
| Feature Vectors        | `data/features/` | 6,000   | Uses a pre-trained ResNet-18 CNN to extract 512 × 1 feature vectors for the RGB images.                                        | `resnet18_test_512.npz`<br> `resnet18_train_512.npz` |
| PCA Features           | `data/features/` | 6,000   | Uses PCA in scikit learn to further reduce the size of feature vectors from 512×1 to 50×1.                                     | `pca_test_50.npz`<br> `pca_train_50.npz`             |
| CIFAR-10 Training Data | `data/training/` | 6,000   | Final training data used for CNN training and evaluation.                                                                      | `cifar10_data.npz`                                   |
| PCA Training Data      | `data/training/` | 6,000   | Final training data used for GNB, DT, and MLP training and evaluation.                                                         | `pca_data.npz`                                       |

## Models

The following models have been implemented and trained on the CIFAR-10 subsets (Note: `{variation}` refers to the specific model variation used):

| Algorithm                               | Dataset Used     | Features             | Location                                                                                                                     |
| --------------------------------------- | ---------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Gaussian Naive Bayes                    | CIFAR-10 Subsets | PCA Features (50-D)  | `models/gnb/custom_gnb.pkl`<br> `models/gnb/sklearn_gnb.pkl`                                                                 |
| Decision Tree Variations                | CIFAR-10 Subsets | PCA Features (50-D)  | `models/decision_tree/{variation}/custom_decision_tree.pkl`<br> `models/decision_tree/{variation}/sklearn_decision_tree.pkl` |
| Multi-Layer Perceptron Variations       | CIFAR-10 Subsets | PCA Features (50-D)  | `models/mlp/{variation}/custom_mlp.pt`<br> `models/mlp/{variation}/sklearn_mlp.pkl`                                          |
| Convolutional Neural Network Variations | CIFAR-10 Subsets | Raw Images (32x32x3) | `models/cnn/{variation}/custom_cnn.pt`                                                                                       |

## Setup Instructions

### Prerequisites

- Python 3.7+
- PyTorch
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
3. **Select model configurations:**
   - Modify the configuration files in each model directory (`models/decision_tree/configs.py`, `models/mlp/configs.py`, `models/cnn/configs.py`) to select desired model variations and hyperparameters.
4. **Train the models**
   ```bash
   python -m models.gnb.train
   python -m models.decision_tree.train
   python -m models.mlp.train
   python -m models.cnn.train
   ```
5. **Evaluate the models**
   ```bash
   python -m models.gnb.evaluate
   python -m models.decision_tree.evaluate
   python -m models.mlp.evaluate
   python -m models.cnn.evaluate
   ```

## Results

The results from the evaluations are stored in the following structure:

Each result directory contains:

- `<model>_metrics.json` – performance metrics (accuracy, training accuracy, precision, recall, F1, train time)
- `<model>_confusion_matrix.png` – confusion matrix
  with the exception of MLP and CNN models which also include:
- `epoch_metrics.txt` – epoch metrics for training and validation accuracy/loss per epoch

```
📂 results/
├── 📂 gnb/
│   ├── 📂 gnb_custom/
│   └── 📂 gnb_sklearn/
│
├── 📂 decision_tree/
│   ├── 📂 decision_tree_custom/
│   │   ├── 📂 depth_5/
│   │   ├── 📂 depth_10/
│   │   ├── 📂 depth_20/
│   │   └── 📂 depth_50/
│   │
│   └── 📂 decision_tree_sklearn/
│       ├── 📂 depth_5/
│       ├── 📂 depth_10/
│       ├── 📂 depth_20/
│       └── 📂 depth_50/
│
├── 📂 mlp/
│   ├── 📂 mlp_custom/
│   │   ├── 📂 base/
│   │   ├── 📂 wide/
│   │   ├── 📂 deep/
│   │   ├── 📂 shallow/
│   │   └── 📂 single/
│   │
│   └── 📂 mlp_sklearn/
│       ├── 📂 base/
│       ├── 📂 wide/
│       ├── 📂 deep/
│       ├── 📂 shallow/
│       └── 📂 single/
│
└── 📂 cnn/
    └── 📂 custom_cnn/
        ├── 📂 shallow/
        ├── 📂 vgg11/
        └── 📂 vgg11_large_kernels/
```

## Report

The full analysis of the models and findings can be found in:

```
report/final_report.pdf
```

## Contributors

- **Matteo Robidoux**
