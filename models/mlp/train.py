import numpy as np
from models.mlp.mlp import MLP
from sklearn.neural_network import MLPClassifier
import time
import os
import joblib
import torch
from tools.evaluation_utils import set_seed
import json
from models.mlp.configs import models, selected_model, learning_rate, epochs

# Set random seed for reproducibility
set_seed(42)

# Paths
custom_model_path = f"./models/mlp/{selected_model}/custom_mlp.pt"
mlp_sklearn_model_path = f"./models/mlp/{selected_model}/sklearn_mlp.pkl"
custom_model_results_path = f"./results/mlp/mlp_custom/{selected_model}/"
mlp_sklearn_results_path = f"./results/mlp/mlp_sklearn/{selected_model}/"
training_data_path = "./data/training/pca_data.npz"

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(mlp_sklearn_results_path, exist_ok=True)

os.makedirs(os.path.dirname(custom_model_path), exist_ok=True)
os.makedirs(os.path.dirname(mlp_sklearn_model_path), exist_ok=True)

# Load saved PCA features
print("Loading saved PCA features...\n")
pca_data = np.load(training_data_path)
X_train, y_train = pca_data['X_train'], pca_data['y_train']
X_test, y_test = pca_data['X_test'], pca_data['y_test']
class_names = pca_data['class_names']

# Train custom MLP
print("Training Custom MLP with " + selected_model + " architecture...\n")
start = time.time()
mlp = MLP(learning_rate=learning_rate, epochs=epochs, layers=models[selected_model]["model"])
mlp.fit(X_train, y_train, output_dir=custom_model_results_path)
custom_mlp_train_time = time.time() - start
torch.save(mlp.state_dict(), custom_model_path)
with open(custom_model_results_path + "custom_mlp_metrics.json", "w") as f:
    json.dump({"train_time": custom_mlp_train_time}, f)
print("\nTraining of Custom MLP completed.\nModel saved to {}\n".format(custom_model_path))

# Train sklearn MLP
print("Training Sklearn MLP with " + selected_model + " architecture...\n")
start = time.time()
sk_mlp = MLPClassifier(hidden_layer_sizes=tuple(models[selected_model]["layers"][1:-1]), activation='relu', solver='sgd', momentum=0.9, learning_rate_init=learning_rate, batch_size=32, max_iter=epochs, random_state=42)
sk_mlp.fit(X_train, y_train)
skl_mlp_train_time = time.time() - start
joblib.dump(sk_mlp, mlp_sklearn_model_path)
with open(mlp_sklearn_results_path + "sklearn_mlp_metrics.json", "w") as f:
    json.dump({"train_time": skl_mlp_train_time}, f)
print("Training of Sklearn MLP completed.\nModel saved to {}\n".format(mlp_sklearn_model_path))