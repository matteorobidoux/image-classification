import numpy as np
import torch
from models.mlp.mlp import MLP
from tools.evaluation_utils import evaluate_model
import joblib
import json
import os
from tools.evaluation_utils import set_seed
from models.mlp.configs import models, selected_model, learning_rate, epochs

# Set seed for reproducibility
set_seed(42)

# Paths
custom_model_path = f"./models/mlp/{selected_model}/custom_mlp.pt"
mlp_sklearn_model_path = f"./models/mlp/{selected_model}/sklearn_mlp.pkl"
custom_model_results_path = f"./results/mlp/mlp_custom/{selected_model}/"
mlp_sklearn_results_path = f"./results/mlp/mlp_sklearn/{selected_model}/"
training_data_path = "./data/training/pca_data.npz"

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(mlp_sklearn_results_path, exist_ok=True)

# Load training set for evaluation
print("Loading test set for evaluation...\n")
test_data = np.load(training_data_path)
X_test, y_test, X_train, y_train, class_names = test_data['X_test'], test_data['y_test'], test_data['X_train'], test_data['y_train'], test_data['class_names']

# Load trained custom MLP
print("Loading trained " + selected_model + " Custom MLP...\n")
mlp = MLP(learning_rate=learning_rate, epochs=epochs, layers=models[selected_model]["model"])
mlp.load_state_dict(torch.load(custom_model_path))
mlp.eval()
with open(custom_model_results_path + "custom_mlp_metrics.json", "r") as f:
    metrics = json.load(f)
    custom_mlp_train_time = metrics['train_time']

# Load trained sklearn MLP
print("Loading trained " + selected_model + " Sklearn MLP...\n")
sk_mlp = joblib.load(mlp_sklearn_model_path)
with open(mlp_sklearn_results_path + "sklearn_mlp_metrics.json", "r") as f:
    metrics = json.load(f)
    skl_mlp_train_time = metrics['train_time']

# Evaluate
print("Evaluating Custom MLP...\n")
evaluate_model(mlp, X_test, y_test, X_train, y_train, class_names, "Custom MLP", custom_model_results_path, custom_mlp_train_time)
print("Custom MLP evaluation completed.\nResults saved to {}\n".format(custom_model_results_path) + '\nEvaluating Sklearn MLP...\n')
evaluate_model(sk_mlp, X_test, y_test, X_train, y_train, class_names, "Sklearn MLP", mlp_sklearn_results_path, skl_mlp_train_time)
print("Sklearn MLP evaluation completed.\nResults saved to {}\n".format(mlp_sklearn_results_path))