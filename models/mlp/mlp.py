import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

class MLP(nn.Module):
    def __init__(self, learning_rate, epochs):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model architecture (Input -> Hidden Layers -> Output)
        self.model = nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ).to(self.device)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def fit(self, X, y, output_dir):
        """Trains the MLP model"""

        # Numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Create TensorDataset and DataLoader to handle batching (32 for SGD)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        self.epoch_data = []

        # Training loop
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            # Iterate over batches
            for inputs, labels in dataloader:

                # Feedforward -> Compute loss -> Backpropagation and optimization -> Update weights
    
                # Reset gradients
                self.optimizer.zero_grad() 
                # Forward pass
                outputs = self.model(inputs)
                # Compute loss
                loss = self.criterion(outputs, labels)
                # Backward pass
                loss.backward()
                # Update weights
                self.optimizer.step()

                # Epoch metrics
                running_loss += loss.item()
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100 * correct / total
            epoch_str = f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%"
            print(epoch_str)

            self.epoch_data.append(epoch_str)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "epoch_metrics.txt"), "w") as f:
            f.write("\n".join(self.epoch_data))
        return self
    
    def predict(self, X):
        """Predicts class labels for input data X"""

        # Feedforward pass
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted = torch.argmax(outputs, dim=1)
        return predicted.cpu().numpy()