import numpy as np
import torch
from models.cnn.cnn import VGG11
from torchvision import datasets
import time
import os
import json

custom_model_path = "./models/cnn/vgg11_custom.pt"
custom_model_results_path = "./results/cnn/vgg11_custom/"
subset_train_path = './data/subsets/cifar10_train_500.pt'
subset_test_path = './data/subsets/cifar10_test_100.pt'
saved_subset_path = './data/subsets/cifar10_saved.npz'

os.makedirs(custom_model_results_path, exist_ok=True)

# Load or save subset data
if not os.path.exists(saved_subset_path):
    print("Loading CIFAR-10 subsets...\n")
    train_data = torch.load(subset_train_path, weights_only=False)
    test_data = torch.load(subset_test_path, weights_only=False)

    print("Preparing training and test data...\n")
    X_train_list, y_train_list = [], []
    for img, label in train_data:
        X_train_list.append(img)
        y_train_list.append(label)
    X_train = torch.stack(X_train_list).numpy()
    y_train = np.array(y_train_list)

    X_test_list, y_test_list = [], []
    for img, label in test_data:
        X_test_list.append(img)
        y_test_list.append(label)
    X_test = torch.stack(X_test_list).numpy()
    y_test = np.array(y_test_list)

    # Retrieve class names from CIFAR-10 dataset
    print("Retrieving CIFAR-10 class names...\n")
    train_dataset = datasets.CIFAR10(root='./data/raw', train=True, download=False)
    class_names = train_dataset.classes

    print("Saving subset data for future use...\n")
    np.savez(saved_subset_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, class_names=class_names)
else:
    print("Loading saved CIFAR-10 subsets...\n")
    data = np.load(saved_subset_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    class_names = data['class_names']

# Train custom CNN
print("Training Custom VGG11 CNN...\n")
start = time.time()
cnn = VGG11(learning_rate=0.01, epochs=10)
cnn.fit(X_train, y_train, output_dir=custom_model_results_path)
train_time = time.time() - start

# Save model and training time
torch.save(cnn.state_dict(), custom_model_path)
with open(os.path.join(custom_model_results_path, "custom_vgg11_cnn_metrics.json"), "w") as f:
    json.dump({"train_time": train_time}, f)

print(f"\nTraining completed. Model saved to {custom_model_path}\n")