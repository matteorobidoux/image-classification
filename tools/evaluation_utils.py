import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import json
import numpy as np
import torch
import random

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar=True)
    
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def evaluate_model(model, X_test, y_test, X_train, y_train, classes, model_name, output_dir, train_time):
    """Evaluates the model and saves metrics and confusion matrix."""
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Calculate metrics
    accuracy = round(accuracy_score(y_test, y_pred_test), 2)
    train_accuracy = round(accuracy_score(y_train, y_pred_train), 2)
    precision = round(precision_score(y_test, y_pred_test, average='weighted', zero_division=0), 2)
    recall = round(recall_score(y_test, y_pred_test, average='weighted', zero_division=0), 2)
    f1 = round(f1_score(y_test, y_pred_test, average='weighted', zero_division=0), 2)
    train_time = round(train_time, 2)
    
    eval_metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'train_accuracy': train_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time
    }
    
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, f"{to_snake_case(model_name)}_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(eval_metrics, f, indent=4)
    
    cm_file = os.path.join(output_dir, f"{to_snake_case(model_name)}_confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred_test, classes, f"{model_name} Confusion Matrix", cm_file)
    
    return eval_metrics

def to_snake_case(name: str) -> str:
    """Convert a string like 'Custom MLP' to 'custom_mlp'."""
    return name.strip().lower().replace(" ", "_")

def set_seed(seed):
    """Sets seeds for reproducible results across numpy, torch, and python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False