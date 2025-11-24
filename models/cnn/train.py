import numpy as np
import torch
from models.cnn.cnn import CNN
import time
import os
import json
from models.cnn.configs import models, selected_model, learning_rate, epochs

custom_model_path = "./models/cnn/" + selected_model + "/custom_cnn.pt"
custom_model_results_path = "./results/cnn/" + selected_model + "/custom_cnn/"
training_data_path = './data/training/cifar10_data.npz'

os.makedirs(custom_model_results_path, exist_ok=True)
os.makedirs(os.path.dirname(custom_model_path), exist_ok=True)

print("Loading saved CIFAR-10 subsets...\n")
data = np.load(training_data_path)
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']
class_names = data['class_names']

# Train custom CNN
print("Training Custom CNN...\n")
start = time.time()
cnn = CNN(learning_rate=learning_rate, epochs=epochs, layers=models[selected_model])
cnn.fit(X_train, y_train, output_dir=custom_model_results_path)
train_time = time.time() - start

# Save model and training time
torch.save(cnn.state_dict(), custom_model_path)
with open(os.path.join(custom_model_results_path, "custom_cnn_metrics.json"), "w") as f:
    json.dump({"train_time": train_time}, f)

print(f"\nTraining completed. Model saved to {custom_model_path}\n")