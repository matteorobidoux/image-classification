import numpy as np
from models.decision_tree.decision_tree import DecisionTree
from tools.evaluation_utils import evaluate_model
import joblib
import json
import pickle
import os
from models.decision_tree.configs import max_depth

# Paths
custom_model_path = "./models/decision_tree/" + "depth_" + str(max_depth) + "/custom_decision_tree.pkl"
decision_tree_sklearn_model_path = "./models/decision_tree/" + "depth_" + str(max_depth) + "/sklearn_decision_tree.pkl"
custom_model_results_path = "./results/decision_tree/decision_tree_custom/" + "depth_" + str(max_depth) + "/"
decision_tree_sklearn_results_path = "./results/decision_tree/decision_tree_sklearn/" + "depth_" + str(max_depth) + "/"
training_data_path = "./data/training/pca_data.npz"

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(decision_tree_sklearn_results_path, exist_ok=True)

# Load training set for evaluation
print("Loading test set for evaluation...\n")
test_data = np.load(training_data_path)
X_test, y_test, X_train, y_train, class_names = test_data['X_test'], test_data['y_test'], test_data['X_train'], test_data['y_train'], test_data['class_names']

# Load trained custom Decision Tree
print("Loading trained Custom Decision Tree with max_depth={}...\n".format(max_depth))
decision_tree = DecisionTree(max_depth=max_depth)
with open(custom_model_path, "rb") as f:
    decision_tree = pickle.load(f)
with open(custom_model_results_path + "custom_decision_tree_metrics.json", "r") as f:
    metrics = json.load(f)
    custom_decision_tree_train_time = metrics['train_time']

# Load trained sklearn Decision Tree
print("Loading trained Sklearn Decision Tree with max_depth={}...\n".format(max_depth))
sk_decision_tree = joblib.load(decision_tree_sklearn_model_path)
with open(decision_tree_sklearn_results_path + "sklearn_decision_tree_metrics.json", "r") as f:
    metrics = json.load(f)
    skl_decision_tree_train_time = metrics['train_time']

# Evaluate
print("Evaluating Custom Decision Tree...\n")
evaluate_model(decision_tree, X_test, y_test, X_train, y_train, class_names, "Custom Decision Tree", custom_model_results_path, custom_decision_tree_train_time)
print("Custom Decision Tree evaluation completed.\nResults saved to {}\n".format(custom_model_results_path) + '\nEvaluating Sklearn Decision Tree...\n')
evaluate_model(sk_decision_tree, X_test, y_test, X_train, y_train, class_names, "Sklearn Decision Tree", decision_tree_sklearn_results_path, skl_decision_tree_train_time)
print("Sklearn Decision Tree evaluation completed.\nResults saved to {}\n".format(decision_tree_sklearn_results_path))