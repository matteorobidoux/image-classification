import numpy as np
import torch
from torchvision import datasets
from tools.cifar10_utils import load_cifar10_data, select_sample, save_dataset
from tools.feature_extraction import (
    get_resnet18_transform,
    load_resnet18_extractor,
    extract_features,
    save_features
)
from tools.pca_utils import apply_pca, save_pca_features
import os

subset_data_path = './data/subsets/'
training_data_path = './data/training/'
features_data_path = './data/features/'

print("Loading CIFAR-10 data...\n")
train_data, test_data = load_cifar10_data()

print("Selecting 500 training and 100 test samples per class...\n")
train_subset = select_sample(train_data, 500)
test_subset = select_sample(test_data, 100)

print("Saving subsets...")
save_dataset(train_subset, subset_data_path + 'cifar10_train_500.pt')
save_dataset(test_subset, subset_data_path + 'cifar10_test_100.pt')
print("Subsets saved to " + subset_data_path + "\n")

print("Retrieving CIFAR-10 class names...\n")
train_dataset = datasets.CIFAR10(root='./data/raw', train=True, download=False)
class_names = train_dataset.classes

print("Preparing CIFAR-10 subsets for training...\n")
X_train_list = []
y_train_list = []

for image, label in train_subset:
    X_train_list.append(image)  
    y_train_list.append(label) 

X_train = torch.stack(X_train_list).numpy()
y_train = np.array(y_train_list)

X_test_list = []
y_test_list = []

for image, label in test_subset:
    X_test_list.append(image)
    y_test_list.append(label)

X_test = torch.stack(X_test_list).numpy()
y_test = np.array(y_test_list)

print("Saving CIFAR-10 subsets for training...")

os.makedirs(training_data_path, exist_ok=True)
np.savez(
    training_data_path + 'cifar10_data.npz',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    class_names=class_names
)
print("CIFAR-10 training subsets saved to " + training_data_path + "cifar10_data.npz\n")

print("Extracting ResNet18 features...\n")
transform = get_resnet18_transform()
train_subset.dataset.transform = transform
test_subset.dataset.transform = transform

model = load_resnet18_extractor()

train_features, train_labels = extract_features(model, train_subset)
test_features, test_labels = extract_features(model, test_subset)

print("Saving ResNet18 features...")
save_features(train_features, train_labels, features_data_path + 'resnet18_train_512.npz')
save_features(test_features, test_labels, features_data_path + 'resnet18_test_512.npz')
print("ResNet18 features saved to " + features_data_path + "\n")

print("Applying PCA to reduce feature dimensions from 512 → 50...\n")
train_pca, test_pca, pca_model = apply_pca(train_features, test_features)

print("Saving PCA features...")
save_pca_features(train_pca, train_labels, features_data_path + 'pca_train_50.npz')
save_pca_features(test_pca, test_labels, features_data_path + 'pca_test_50.npz')
print("PCA features saved to " + features_data_path + "\n")

print("Saving PCA dataset for training...")
np.savez(
    training_data_path + 'pca_data.npz',
    X_train=train_pca,
    y_train=train_labels,
    X_test=test_pca,
    y_test=test_labels,
    class_names=class_names
)
print("PCA training dataset saved to " + training_data_path + "pca_data.npz\n")
print("Preprocessing completed.\n")