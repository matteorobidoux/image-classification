import numpy as np
from models.decision_tree.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
import time
import os
import joblib
import pickle
import json
from models.decision_tree.configs import max_depth

# Paths
custom_model_path = "./models/decision_tree/" + "depth_" + str(max_depth) + "/custom_decision_tree.pkl"
decision_tree_sklearn_model_path = "./models/decision_tree/" + "depth_" + str(max_depth) + "/sklearn_decision_tree.pkl"
custom_model_results_path = "./results/decision_tree/decision_tree_custom/" + "depth_" + str(max_depth) + "/"
decision_tree_sklearn_results_path = "./results/decision_tree/decision_tree_sklearn/" + "depth_" + str(max_depth) + "/"
training_data_path = "./data/training/pca_data.npz"

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(decision_tree_sklearn_results_path, exist_ok=True)
os.makedirs(os.path.dirname(custom_model_path), exist_ok=True)
os.makedirs(os.path.dirname(decision_tree_sklearn_model_path), exist_ok=True)

# Load saved PCA features
print("Loading saved PCA features...\n")
pca_data = np.load(training_data_path)
X_train, y_train = pca_data['X_train'], pca_data['y_train']
X_test, y_test = pca_data['X_test'], pca_data['y_test']
class_names = pca_data['class_names']

# Train custom Decision Tree
print("Training Custom Decision Tree with max_depth={}...\n".format(max_depth))
start = time.time()
decision_tree = DecisionTree(max_depth=max_depth)
decision_tree.fit(X_train, y_train)
custom_decision_tree_train_time = time.time() - start
with open(custom_model_path, "wb") as f:
    pickle.dump(decision_tree, f)
with open(custom_model_results_path + "custom_decision_tree_metrics.json", "w") as f:
    json.dump({"train_time": custom_decision_tree_train_time}, f)
print("\nTraining of Custom Decision Tree completed.\nModel saved to {}\n".format(custom_model_path))

# Train sklearn Decision Tree
print("Training Sklearn Decision Tree...\n")
start = time.time()
sk_decision_tree = DecisionTreeClassifier(max_depth=max_depth, criterion='gini', random_state=42)
sk_decision_tree.fit(X_train, y_train)
skl_decision_tree_train_time = time.time() - start
joblib.dump(sk_decision_tree, decision_tree_sklearn_model_path)
with open(decision_tree_sklearn_results_path + "sklearn_decision_tree_metrics.json", "w") as f:
    json.dump({"train_time": skl_decision_tree_train_time}, f)
print("Training of Sklearn Decision Tree completed.\nModel saved to {}\n".format(decision_tree_sklearn_model_path))