import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# Importa as classes de otimizador para a checagem 'isinstance'
from optimizers import SIGMA_D, SIGMA_C

# ============================================================================
# FUNÇÕES DE DADOS 
# ============================================================================

def get_data_loaders(batch_size=128):
    """Carrega os datasets MNIST para treino e teste."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# ============================================================================
# FUNÇÕES DE TREINAMENTO E AVALIAÇÃO 
# ============================================================================

def train_epoch(model, optimizer, train_loader, device, loss_fn):
    """Treina o modelo por uma época."""
    
    model.train()
    total_loss = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        loss_item = loss.item()

        # Passa loss_item se for SIGMA-D ou SIGMA-C
        if isinstance(optimizer, (SIGMA_D, SIGMA_C)):
            optimizer.step(loss_item=loss_item) 
        else:
            optimizer.step()
        
        total_loss += loss_item
        
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, loss_fn):
    """Avalia o modelo no conjunto de teste."""
    
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            test_loss += loss.item() * data.size(0) 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy

# ============================================================================
# FUNÇÃO PARA EXECUTAR EXPERIMENTOS 
# ============================================================================

def run_experiment(experiment_name, model, optimizer_config, train_loader, test_loader, 
                   device, loss_fn, n_epochs):
    """
    Executa um experimento de treinamento.
    """
    
    print("\n" + "="*80)
    print(f"TREINANDO: {experiment_name}")
    print("="*80)
    
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    start_time = time.time()
    epoch_counter = 0
    
    for phase_idx, (optimizer, phase_epochs) in enumerate(optimizer_config):
        phase_name = f"Fase {phase_idx + 1}" if len(optimizer_config) > 1 else "Única"
        print(f"\n--- {phase_name}: {optimizer.__class__.__name__} ---")
        
        for epoch in range(phase_epochs):
            train_loss = train_epoch(model, optimizer, train_loader, device, loss_fn)
            test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            epoch_counter += 1
            print(f"Época [{epoch_counter:2d}/{n_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\n--- {experiment_name} concluído em {elapsed_time:.2f}s ---")
    
    return history, elapsed_time