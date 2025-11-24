import numpy as np
from models.gnb.gnb import GaussianNaiveBayes
from tools.evaluation_utils import evaluate_model
import joblib
import json
import pickle
import os

# Paths
custom_model_path = "./models/gnb/custom_gnb.pkl"
gnb_sklearn_model_path = "./models/gnb/sklearn_gnb.pkl"
custom_model_results_path = "./results/gnb/gnb_custom/"
gnb_sklearn_results_path = "./results/gnb/gnb_sklearn/"

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(gnb_sklearn_results_path, exist_ok=True)

# Load training set for evaluation
print("Loading test set for evaluation...\n")
test_data = np.load("./data/training/pca_data.npz")
X_test, y_test, X_train, y_train, class_names = test_data['X_test'], test_data['y_test'], test_data['X_train'], test_data['y_train'], test_data['class_names']

# Load trained custom GNB
print("Loading trained Custom GNB...\n")
gnb = GaussianNaiveBayes()
with open(custom_model_path, "rb") as f:
    gnb = pickle.load(f)
with open(custom_model_results_path + "custom_gnb_metrics.json", "r") as f:
    metrics = json.load(f)
    custom_gnb_train_time = metrics['train_time']

# Load trained sklearn GNB
print("Loading trained Sklearn GNB...\n")
sk_gnb = joblib.load(gnb_sklearn_model_path)
with open(gnb_sklearn_results_path + "sklearn_gnb_metrics.json", "r") as f:
    metrics = json.load(f)
    skl_gnb_train_time = metrics['train_time']

# Evaluate
print("Evaluating Custom GNB...\n")
evaluate_model(gnb, X_test, y_test, X_train, y_train, class_names, "Custom GNB", custom_model_results_path, custom_gnb_train_time)
print("Custom GNB evaluation completed.\nResults saved to {}\n".format(custom_model_results_path) + '\nEvaluating Sklearn GNB...\n')
evaluate_model(sk_gnb, X_test, y_test, X_train, y_train, class_names, "Sklearn GNB", gnb_sklearn_results_path, skl_gnb_train_time)
print("Sklearn GNB evaluation completed.\nResults saved to {}\n".format(gnb_sklearn_results_path))