import numpy as np
from models.gnb.gnb import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
import time
import os
import joblib
import pickle
import json

# Paths
custom_model_path = "./models/gnb/custom_gnb.pkl"
gnb_sklearn_model_path = "./models/gnb/sklearn_gnb.pkl"
custom_model_results_path = "./results/gnb/gnb_custom/"
gnb_sklearn_results_path = "./results/gnb/gnb_sklearn/"
training_data_path = './data/training/pca_data.npz'

# Ensure directories exist
os.makedirs(os.path.dirname(custom_model_path), exist_ok=True)
os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(gnb_sklearn_results_path, exist_ok=True)

# Load saved PCA features
print("Loading saved PCA features...\n")
pca_data = np.load(training_data_path)
X_train, y_train = pca_data['X_train'], pca_data['y_train']
X_test, y_test = pca_data['X_test'], pca_data['y_test']
class_names = pca_data['class_names']

# Train custom GNB
print("Training Custom GNB...\n")
start = time.time()
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)
custom_gnb_train_time = time.time() - start
with open(custom_model_path, "wb") as f:
    pickle.dump(gnb, f)
with open(custom_model_results_path + "custom_gnb_metrics.json", "w") as f:
    json.dump({"train_time": custom_gnb_train_time}, f)
print("\nTraining of Custom GNB completed.\nModel saved to {}\n".format(custom_model_path))

# Train sklearn GNB
print("Training Sklearn GNB...\n")
start = time.time()
sk_gnb = GaussianNB()
sk_gnb.fit(X_train, y_train)
skl_gnb_train_time = time.time() - start
joblib.dump(sk_gnb, gnb_sklearn_model_path)
with open(gnb_sklearn_results_path + "sklearn_gnb_metrics.json", "w") as f:
    json.dump({"train_time": skl_gnb_train_time}, f)
print("Training of Sklearn GNB completed.\nModel saved to {}\n".format(gnb_sklearn_model_path))