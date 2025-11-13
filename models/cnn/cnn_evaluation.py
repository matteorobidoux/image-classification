import numpy as np
import torch
from models.cnn.cnn import VGG11
from tools.evaluation_utils import evaluate_model
import json
import os

custom_model_path = "./models/cnn/vgg11_custom.pt"
custom_model_results_path = "./results/cnn/vgg11_custom/"
saved_subset_path = './data/subsets/cifar10_saved.npz'

os.makedirs(custom_model_results_path, exist_ok=True)

# Load saved test data
print("Loading saved CIFAR-10 subsets for evaluation...\n")
data = np.load(saved_subset_path)
X_test, y_test, class_names = data['X_test'], data['y_test'], data['class_names']

# Load trained CNN
print("Loading trained Custom VGG11 CNN...\n")
cnn = VGG11(learning_rate=0.01, epochs=10)
cnn.load_state_dict(torch.load(custom_model_path))

# Load training metrics
with open(os.path.join(custom_model_results_path, "custom_vgg11_cnn_metrics.json"), "r") as f:
    metrics = json.load(f)
    train_time = metrics["train_time"]

# Evaluate
print("Evaluating Custom VGG11 CNN...\n")
evaluate_model(cnn, X_test, y_test, class_names, "Custom VGG11 CNN", custom_model_results_path, train_time)
print(f"Evaluation completed. Results saved to {custom_model_results_path}\n")
