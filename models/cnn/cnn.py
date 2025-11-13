import torch
import torch.nn as nn
import torch.optim as optim
import os

class VGG11(nn.Module):
    def __init__(self, learning_rate, epochs):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convolutional and Fully Connected layers for VGG11 architecture
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 10)
        )

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        
        self.to(self.device)

    def forward(self, x):
        """Defines the forward pass of the VGG11 model"""
        x = x.to(self.device)
        # Convolutional + Pooling layers extraction
        x = self.convolutional_layer(x)
        # Flatten 2D feature maps to 1D feature vectors
        x = x.view(x.size(0), -1)
        # Fully connected layers for classification
        x = self.fully_connected_layer(x)
        return x

    def fit(self, X, y, output_dir):
        """Trains the VGG11 model"""

        # Numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Create DataLoader to handle batching (32 for SGD)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

        self.train()
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
                outputs = self.forward(inputs)
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
        self.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            predicted = torch.argmax(outputs, dim=1)
        return predicted.cpu().numpy()