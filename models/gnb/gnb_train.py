import numpy as np
from models.gnb.gnb import GaussianNaiveBayes
from sklearn.naive_bayes import GaussianNB
from torchvision import datasets
import time
import os
import joblib
import pickle
import json

custom_model_path = "./models/gnb/custom_gnb.pkl"
gnb_sklearn_model_path = "./models/gnb/sklearn_gnb.pkl"
custom_model_results_path = "./results/gnb/gnb_custom/"
gnb_sklearn_results_path = "./results/gnb/gnb_sklearn/"
pca_data_path = './data/features/pca'

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(gnb_sklearn_results_path, exist_ok=True)

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