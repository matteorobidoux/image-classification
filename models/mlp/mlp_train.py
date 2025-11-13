import numpy as np
from models.mlp.mlp import MLP
from sklearn.neural_network import MLPClassifier
from torchvision import datasets
import time
import os
import joblib
import torch
import json

custom_model_path = "./models/mlp/custom_mlp.pt"
mlp_sklearn_model_path = "./models/mlp/sklearn_mlp.pkl"
custom_model_results_path = "./results/mlp/mlp_custom/"
mlp_sklearn_results_path = "./results/mlp/mlp_sklearn/"
pca_data_path = './data/features/pca'

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(mlp_sklearn_results_path, exist_ok=True)

if not os.path.exists(pca_data_path + '_saved.npz'):
    # Load PCA features
    print("Loading PCA features...\n")
    train_data = np.load(pca_data_path + '_train_50.npz')
    test_data = np.load(pca_data_path + '_test_50.npz')

    X_train, y_train = train_data['features'], train_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']

    # Retrieve class names from CIFAR-10 dataset
    print("Retrieving CIFAR-10 class names...\n")
    train_dataset = datasets.CIFAR10(root='./data/raw', train=True, download=False)
    class_names = train_dataset.classes

    # Save loaded PCA features for evaluation use
    print("Saving loaded PCA features for evaluation use...\n")
    np.savez(pca_data_path + '_saved.npz', X_test=X_test, y_test=y_test, class_names=class_names, X_train=X_train, y_train=y_train)
else:
    # Load saved PCA features
    print("Loading saved PCA features...\n")
    pca_data = np.load(pca_data_path + '_saved.npz')
    X_train, y_train = pca_data['X_train'], pca_data['y_train']
    X_test, y_test = pca_data['X_test'], pca_data['y_test']
    class_names = pca_data['class_names']

# Train custom MLP
print("Training Custom MLP...\n")
start = time.time()
mlp = MLP(learning_rate=0.001, epochs=100)
mlp.fit(X_train, y_train, output_dir=custom_model_results_path)
custom_mlp_train_time = time.time() - start
torch.save(mlp.state_dict(), custom_model_path)
with open(custom_model_results_path + "custom_mlp_metrics.json", "w") as f:
    json.dump({"train_time": custom_mlp_train_time}, f)
print("\nTraining of Custom MLP completed.\nModel saved to {}\n".format(custom_model_path))

# Train sklearn MLP
print("Training Sklearn MLP...\n")
start = time.time()
sk_mlp = MLPClassifier(hidden_layer_sizes=(512,512), activation='relu', solver='sgd', momentum=0.9, learning_rate_init=0.001, batch_size=32, max_iter=100, random_state=42)
sk_mlp.fit(X_train, y_train)
skl_mlp_train_time = time.time() - start
joblib.dump(sk_mlp, mlp_sklearn_model_path)
with open(mlp_sklearn_results_path + "sklearn_mlp_metrics.json", "w") as f:
    json.dump({"train_time": skl_mlp_train_time}, f)
print("Training of Sklearn MLP completed.\nModel saved to {}\n".format(mlp_sklearn_model_path))