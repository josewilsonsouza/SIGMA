import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """Rede Neural Feedforward simples para MNIST (784 → 128 → 64 → 10)."""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LogisticRegression(nn.Module):
    """Regressão Logística para MNIST (784 → 10)."""
    
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(784, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        return self.linear(x)