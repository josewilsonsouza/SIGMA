import torch
import torch.nn as nn
import torch.nn.functional as F

def CReLU(z):
    """
    Ativação CReLU para números complexos.
    Aplica ReLU separadamente às partes real e imaginária.

    CReLU(z) = ReLU(Re(z)) + i * ReLU(Im(z))
    """
    return F.relu(z.real) + 1j * F.relu(z.imag)


class ComplexMNISTNet(nn.Module):
    """Rede Neural Feedforward COMPLEXA (784 → 128 → 64 → 10)."""

    def __init__(self):
        super(ComplexMNISTNet, self).__init__()
        # Camadas lineares agora usam dtype=torch.cfloat
        self.fc1 = nn.Linear(784, 128, dtype=torch.cfloat)
        self.fc2 = nn.Linear(128, 64, dtype=torch.cfloat)
        self.fc3 = nn.Linear(64, 10, dtype=torch.cfloat)

    def forward(self, x):
        # x já é complexo
        x = x.view(-1, 784)
        x = CReLU(self.fc1(x))
        x = CReLU(self.fc2(x))
        return self.fc3(x) # Saída complexa


class ComplexLogisticRegression(nn.Module):
    """Regressão Logística COMPLEXA (784 → 10)."""

    def __init__(self):
        super(ComplexLogisticRegression, self).__init__()
        self.linear = nn.Linear(784, 10, dtype=torch.cfloat)

    def forward(self, x):
        # x já é complexo
        x = x.view(-1, 784)
        return self.linear(x) # Saída complexa
