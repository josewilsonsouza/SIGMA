import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from optimizers_complex import Complex_SIGMA_D, Complex_SIGMA_C


def get_data_loaders(batch_size=128):
    """
    Carrega os datasets MNIST (dados permanecem reais aqui).

    Args:
        batch_size: Tamanho do batch para treinamento

    Returns:
        train_loader: DataLoader para conjunto de treinamento
        test_loader: DataLoader para conjunto de teste
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def train_epoch(model, optimizer, train_loader, device, loss_fn):
    """
    Treina o modelo por uma época (adaptado para redes complexas).

    Args:
        model: Modelo complexo a ser treinado
        optimizer: Otimizador
        train_loader: DataLoader de treinamento
        device: Dispositivo (CPU/CUDA)
        loss_fn: Função de perda

    Returns:
        float: Loss média de treinamento
    """
    model.train()
    total_loss = 0

    for data, target in train_loader:
        # data (REAL) -> data (COMPLEXO)
        data = data.to(device).to(torch.cfloat)
        target = target.to(device) # REAL

        optimizer.zero_grad()
        output = model(data) # COMPLEXO

        # CrossEntropyLoss não aceita input complexo.
        # Usamos a magnitude da saída para calcular a loss.
        loss = loss_fn(output.abs(), target) # REAL

        loss.backward()
        loss_item = loss.item() # REAL

        # Passa loss_item (real) para os otimizadores
        if isinstance(optimizer, (Complex_SIGMA_D, Complex_SIGMA_C)):
            optimizer.step(loss_item=loss_item)
        else:
            optimizer.step()

        total_loss += loss_item

    return total_loss / len(train_loader)


def evaluate(model, test_loader, device, loss_fn):
    """
    Avalia o modelo no conjunto de teste (adaptado para redes complexas).

    Args:
        model: Modelo complexo a ser avaliado
        test_loader: DataLoader de teste
        device: Dispositivo (CPU/CUDA)
        loss_fn: Função de perda

    Returns:
        tuple: (test_loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # data (REAL) -> data (COMPLEXO)
            data = data.to(device).to(torch.cfloat)
            target = target.to(device) # REAL

            output = model(data) # COMPLEXO

            # Usamos a magnitude para loss e predição
            output_real_abs = output.abs()
            loss = loss_fn(output_real_abs, target) # REAL
            test_loss += loss.item() * data.size(0)

            pred = output_real_abs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss, accuracy


def run_experiment(experiment_name, model, optimizer_config, train_loader, test_loader,
                   device, loss_fn, n_epochs):
    """
    Executa um experimento de treinamento completo.

    Args:
        experiment_name: Nome do experimento
        model: Modelo a ser treinado
        optimizer_config: Lista de tuplas (optimizer, num_epochs) para treinamento híbrido
        train_loader: DataLoader de treinamento
        test_loader: DataLoader de teste
        device: Dispositivo (CPU/CUDA)
        loss_fn: Função de perda
        n_epochs: Número total de épocas

    Returns:
        tuple: (history, elapsed_time)
            - history: Dict com 'train_loss', 'test_loss', 'test_acc'
            - elapsed_time: Tempo de execução em segundos
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
