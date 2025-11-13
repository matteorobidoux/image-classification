import numpy as np
from models.decision_tree.decision_tree import DecisionTree
from tools.evaluation_utils import evaluate_model
import joblib
import json
import pickle
import os

custom_model_path = "./models/decision_tree/custom_decision_tree.pkl"
decision_tree_sklearn_model_path = "./models/decision_tree/sklearn_decision_tree.pkl"
custom_model_results_path = "./results/decision_tree/decision_tree_custom/"
decision_tree_sklearn_results_path = "./results/decision_tree/decision_tree_sklearn/"

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(decision_tree_sklearn_results_path, exist_ok=True)

# Load training set for evaluation
print("Loading test set for evaluation...\n")
test_data = np.load("./data/features/pca_saved.npz")
X_test, y_test, class_names = test_data['X_test'], test_data['y_test'], test_data['class_names']

# Load trained custom Decision Tree
print("Loading trained Custom Decision Tree...\n")
decision_tree = DecisionTree(max_depth=50, min_samples_split=5)
with open(custom_model_path, "rb") as f:
    decision_tree = pickle.load(f)
with open(custom_model_results_path + "custom_decision_tree_metrics.json", "r") as f:
    metrics = json.load(f)
    custom_decision_tree_train_time = metrics['train_time']

# Load trained sklearn Decision Tree
print("Loading trained Sklearn Decision Tree...\n")
sk_decision_tree = joblib.load(decision_tree_sklearn_model_path)
with open(decision_tree_sklearn_results_path + "sklearn_decision_tree_metrics.json", "r") as f:
    metrics = json.load(f)
    skl_decision_tree_train_time = metrics['train_time']

# Evaluate
print("Evaluating Custom Decision Tree...\n")
evaluate_model(decision_tree, X_test, y_test, class_names, "Custom Decision Tree", custom_model_results_path, custom_decision_tree_train_time)
print("Custom Decision Tree evaluation completed.\nResults saved to {}\n".format(custom_model_results_path) + '\nEvaluating Sklearn Decision Tree...\n')
evaluate_model(sk_decision_tree, X_test, y_test, class_names, "Sklearn Decision Tree", decision_tree_sklearn_results_path, skl_decision_tree_train_time)
print("Sklearn Decision Tree evaluation completed.\nResults saved to {}\n".format(decision_tree_sklearn_results_path))