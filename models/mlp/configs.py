from torch import nn
# MLP configuration parameters

models = {
    "single": {
        "layers": [50, 10],
        "model": nn.Sequential(
            nn.Linear(50, 10),
        )
    },
    "shallow": {
        "layers": [50, 128, 10],
        "model": nn.Sequential(
            nn.Linear(50, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    },
    "base": {
        "layers": [50, 512, 512, 10],
        "model": nn.Sequential(
            nn.Linear(50, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    },
    "deep": {
        "layers": [50, 512, 512, 512, 512, 10],
        "model": nn.Sequential(
            nn.Linear(50, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    },
    "wide": {
        "layers": [50, 1024, 1024, 10],
        "model": nn.Sequential(
            nn.Linear(50, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
    }
}

selected_model = "wide"
epochs = 100
learning_rate = 0.001
