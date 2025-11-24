import torch
from torchvision import models, transforms
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import os

def get_resnet18_transform():
    """Returns transformation for ResNet18 model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform

def load_resnet18_extractor():
    """Loads pretrained ResNet18 model without classification layer"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

def extract_features(model, data):
    """Extracts features from data using the given model"""
    loader = DataLoader(data, batch_size=32, shuffle=False)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            features = model(images).view(images.size(0), -1)
            all_features.append(features.numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    return all_features, all_labels

def save_features(features, labels, filename):
    """Saves extracted features and labels to a file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, features=features, labels=labels)