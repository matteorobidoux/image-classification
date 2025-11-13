from sklearn.decomposition import PCA
import numpy as np
import os

def apply_pca(train_features, test_features):
    """Reduce feature vectors size from 512 to 50 using PCA."""
    pca = PCA(n_components=50)
    train_transformed = pca.fit_transform(train_features)
    test_transformed = pca.transform(test_features)
    return train_transformed, test_transformed, pca

def save_pca_features(features, labels, filename):
    """Save PCA features and labels"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, features=features, labels=labels)