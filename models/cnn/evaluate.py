import numpy as np
import torch
from models.cnn.cnn import CNN
from tools.evaluation_utils import evaluate_model
import json
import os
from tools.evaluation_utils import set_seed
from models.cnn.configs import selected_model, learning_rate, epochs, models

# Set seed for reproducibility
set_seed(42)

# Paths
custom_model_path = "./models/cnn/" + selected_model + "/custom_cnn.pt"
custom_model_results_path = "./results/cnn/custom_cnn/" + selected_model
training_data_path = './data/training/cifar10_data.npz'

os.makedirs(custom_model_results_path, exist_ok=True)

# Load saved test data
print("Loading saved CIFAR-10 subsets for evaluation...\n")
data = np.load(training_data_path)
X_test, y_test, X_train, y_train, class_names = data['X_test'], data['y_test'], data['X_train'], data['y_train'], data['class_names']

# Load trained CNN
print("Loading trained Custom CNN...\n")
cnn = CNN(learning_rate=learning_rate, epochs=epochs, layers=models[selected_model])
cnn.load_state_dict(torch.load(custom_model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

# Load training metrics
if os.path.exists(os.path.join(custom_model_results_path, "custom_cnn_metrics.json")):
    with open(os.path.join(custom_model_results_path, "custom_cnn_metrics.json"), "r") as f:
        metrics = json.load(f)
        train_time = metrics["train_time"]
else:
    train_time = None

# Evaluate
print("Evaluating Custom CNN...\n")
evaluate_model(cnn, X_test, y_test, X_train, y_train, class_names, "Custom CNN", custom_model_results_path, train_time)
print(f"Evaluation completed. Results saved to {custom_model_results_path}\n")
