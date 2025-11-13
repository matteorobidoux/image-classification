import numpy as np
from models.decision_tree.decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from torchvision import datasets
import time
import os
import joblib
import pickle
import json

custom_model_path = "./models/decision_tree/custom_decision_tree.pkl"
decision_tree_sklearn_model_path = "./models/decision_tree/sklearn_decision_tree.pkl"
custom_model_results_path = "./results/decision_tree/decision_tree_custom/"
decision_tree_sklearn_results_path = "./results/decision_tree/decision_tree_sklearn/"
pca_data_path = './data/features/pca'

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(decision_tree_sklearn_results_path, exist_ok=True)

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

# Train custom Decision Tree
print("Training Custom Decision Tree...\n")
start = time.time()
decision_tree = DecisionTree(max_depth=50, min_samples_split=5)
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
sk_decision_tree = DecisionTreeClassifier(max_depth=50, criterion='gini', random_state=42)
sk_decision_tree.fit(X_train, y_train)
skl_decision_tree_train_time = time.time() - start
joblib.dump(sk_decision_tree, decision_tree_sklearn_model_path)
with open(decision_tree_sklearn_results_path + "sklearn_decision_tree_metrics.json", "w") as f:
    json.dump({"train_time": skl_decision_tree_train_time}, f)
print("Training of Sklearn Decision Tree completed.\nModel saved to {}\n".format(decision_tree_sklearn_model_path))