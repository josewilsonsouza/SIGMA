"""
Benchmark Comparativo de Híbridos Sequenciais no MNIST (VERSÃO COMPLEXA)
======================================================================
Script Principal para Execução

Este script importa:
- Otimizadores de 'optimizers_complex.py'
- Modelos de 'models.py'
- Funções de utilidade de 'utils_complex.py'
- Funções de plotagem de 'plotting_complex.py'

Experimentos (Redes Neurais Complexas):
1. Adam (Baseline): 20 épocas contínuas
2. Híbrido (Adam → SGD+M): 10 épocas Adam + 10 épocas SGD+M
3. Híbrido (Adam → Complex_SIGMA-D): 10 épocas Adam + 10 épocas SIGMA (Score D)
4. Híbrido (Adam → Complex_SIGMA-C): 10 épocas Adam + 10 épocas SIGMA (Score C)

Experimentos (Regressão Logística Complexa):
5. Otimizadores Puros (Adam, SGD, SIGMA-D, SIGMA-C)
6. Híbridos (Adam → SGD, Adam → SIGMA-D, Adam → SIGMA-C)

"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Importações dos arquivos locais
from models import ComplexMNISTNet, ComplexLogisticRegression
from optimizers_complex import Complex_SIGMA_D, Complex_SIGMA_C
from utils_complex import get_data_loaders, run_experiment
from plotting_complex import generate_nn_plots, generate_lr_plots


def main():
    """Executa todos os experimentos complexos e gera análises."""

    # Configuração
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")

    # Configurações de Épocas da Rede Neural
    N_EPOCHS_NN_TOTAL = 20
    N_EPOCHS_NN_PHASE1 = 10
    N_EPOCHS_NN_PHASE2 = N_EPOCHS_NN_TOTAL - N_EPOCHS_NN_PHASE1

    # Configurações de Épocas da Regressão Logística
    N_EPOCHS_LR_TOTAL = 30
    N_EPOCHS_LR_PHASE1 = 15
    N_EPOCHS_LR_PHASE2 = N_EPOCHS_LR_TOTAL - N_EPOCHS_LR_PHASE1

    # Parâmetros de LR (baseados em _SIGMA.py)
    LR_ADAM = 0.001
    LR_SGD = 0.01
    LR_SIGMA = 0.01

    # Dados
    train_loader, test_loader = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()

    # =======================================================================
    # PARTE 1: EXPERIMENTOS COM REDES NEURAIS COMPLEXAS
    # =======================================================================

    print("\n" + "="*80)
    print("PARTE 1: EXPERIMENTOS COM REDES NEURAIS COMPLEXAS")
    print("="*80)

    base_model = ComplexMNISTNet().to(DEVICE)

    results_nn = {}
    times_nn = {}

    # --- EXPERIMENTO 1: Adam (Baseline) ---
    model_adam = copy.deepcopy(base_model)
    # optim.Adam suporta parâmetros complexos nativamente
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=LR_ADAM)

    history_adam, time_adam = run_experiment(
        experiment_name="Complex Adam (Baseline)",
        model=model_adam,
        optimizer_config=[(optimizer_adam, N_EPOCHS_NN_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Complex Adam (Baseline)'] = history_adam
    times_nn['Complex Adam (Baseline)'] = time_adam

    # --- EXPERIMENTO 2: Híbrido (Adam → SGD+M) ---
    model_hybrid_sgd = copy.deepcopy(base_model)
    optimizer_adam_phase1_ctrl = optim.Adam(model_hybrid_sgd.parameters(), lr=LR_ADAM)
    # optim.SGD suporta parâmetros complexos nativamente
    optimizer_sgd_phase2 = optim.SGD(model_hybrid_sgd.parameters(), lr=LR_SGD, momentum=0.9)

    history_hybrid_sgd, time_hybrid_sgd = run_experiment(
        experiment_name="Híbrido (Adam -> Complex SGD+M)",
        model=model_hybrid_sgd,
        optimizer_config=[
            (optimizer_adam_phase1_ctrl, N_EPOCHS_NN_PHASE1),
            (optimizer_sgd_phase2, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> Complex SGD+M'] = history_hybrid_sgd
    times_nn['Adam -> Complex SGD+M'] = time_hybrid_sgd

    # --- EXPERIMENTO 3: Híbrido (Adam → Complex_SIGMA-D) ---
    model_hybrid_sigma_d = copy.deepcopy(base_model)
    optimizer_adam_phase1_d = optim.Adam(model_hybrid_sigma_d.parameters(), lr=LR_ADAM)
    optimizer_sigma_d = Complex_SIGMA_D(
        model_hybrid_sigma_d.parameters(),
        lr=LR_SIGMA,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0
    )

    history_hybrid_sigma_d, time_hybrid_sigma_d = run_experiment(
        experiment_name="Híbrido (Adam -> Complex SIGMA-D)",
        model=model_hybrid_sigma_d,
        optimizer_config=[
            (optimizer_adam_phase1_d, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_d, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> Complex SIGMA-D'] = history_hybrid_sigma_d
    times_nn['Adam -> Complex SIGMA-D'] = time_hybrid_sigma_d

    # --- EXPERIMENTO 4: Híbrido (Adam → Complex_SIGMA-C) ---
    model_hybrid_sigma_c = copy.deepcopy(base_model)
    optimizer_adam_phase1_c = optim.Adam(model_hybrid_sigma_c.parameters(), lr=LR_ADAM)
    optimizer_sigma_c = Complex_SIGMA_C(
        model_hybrid_sigma_c.parameters(),
        lr=LR_SIGMA,
        beta=0.9,
        alpha_min=0.1,
        alpha_max=2.0
    )

    history_hybrid_sigma_c, time_hybrid_sigma_c = run_experiment(
        experiment_name="Híbrido (Adam -> Complex SIGMA-C)",
        model=model_hybrid_sigma_c,
        optimizer_config=[
            (optimizer_adam_phase1_c, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_c, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> Complex SIGMA-C'] = history_hybrid_sigma_c
    times_nn['Adam -> Complex SIGMA-C'] = time_hybrid_sigma_c


    # =======================================================================
    # PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA COMPLEXA
    # =======================================================================

    print("\n" + "="*80)
    print("PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA COMPLEXA")
    print("="*80)

    base_logistic = ComplexLogisticRegression().to(DEVICE)
    results_lr = {}
    times_lr = {}

    # --- Experimento LR-Puro 1: Adam ---
    model_lr_adam = copy.deepcopy(base_logistic)
    optimizer_lr_adam = optim.Adam(model_lr_adam.parameters(), lr=LR_ADAM)
    history_lr_adam, time_lr_adam = run_experiment(
        experiment_name="[LR] Complex Adam (Puro)",
        model=model_lr_adam,
        optimizer_config=[(optimizer_lr_adam, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex Adam (Puro)'] = history_lr_adam
    times_lr['Complex Adam (Puro)'] = time_lr_adam

    # --- Experimento LR-Puro 2: SGD+Momentum ---
    model_lr_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_sgd = optim.SGD(model_lr_sgd.parameters(), lr=LR_SGD, momentum=0.9)
    history_lr_sgd, time_lr_sgd = run_experiment(
        experiment_name="[LR] Complex SGD+M (Puro)",
        model=model_lr_sgd,
        optimizer_config=[(optimizer_lr_sgd, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex SGD+M (Puro)'] = history_lr_sgd
    times_lr['Complex SGD+M (Puro)'] = time_lr_sgd

    # --- Experimento LR-Puro 3: Complex_SIGMA-D ---
    model_lr_sigma_d = copy.deepcopy(base_logistic)
    optimizer_lr_sigma_d = Complex_SIGMA_D(
        model_lr_sigma_d.parameters(), lr=LR_SIGMA, beta=0.9
    )
    history_lr_sigma_d, time_lr_sigma_d = run_experiment(
        experiment_name="[LR] Complex SIGMA-D (Puro)",
        model=model_lr_sigma_d,
        optimizer_config=[(optimizer_lr_sigma_d, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex SIGMA-D (Puro)'] = history_lr_sigma_d
    times_lr['Complex SIGMA-D (Puro)'] = time_lr_sigma_d

    # --- Experimento LR-Puro 4: Complex_SIGMA-C ---
    model_lr_sigma_c = copy.deepcopy(base_logistic)
    optimizer_lr_sigma_c = Complex_SIGMA_C(
        model_lr_sigma_c.parameters(), lr=LR_SIGMA, beta=0.9
    )
    history_lr_sigma_c, time_lr_sigma_c = run_experiment(
        experiment_name="[LR] Complex SIGMA-C (Puro)",
        model=model_lr_sigma_c,
        optimizer_config=[(optimizer_lr_sigma_c, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Complex SIGMA-C (Puro)'] = history_lr_sigma_c
    times_lr['Complex SIGMA-C (Puro)'] = time_lr_sigma_c

    # --- Experimento LR-Híbrido 1: Adam -> SGD+M ---
    model_lr_h_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam = optim.Adam(model_lr_h_sgd.parameters(), lr=LR_ADAM)
    optimizer_lr_h_sgd = optim.SGD(model_lr_h_sgd.parameters(), lr=LR_SGD, momentum=0.9)

    history_lr_h_sgd, time_lr_h_sgd = run_experiment(
        experiment_name="[LR] Adam -> Complex SGD+M",
        model=model_lr_h_sgd,
        optimizer_config=[
            (optimizer_lr_h_adam, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sgd, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> Complex SGD+M'] = history_lr_h_sgd
    times_lr['Adam -> Complex SGD+M'] = time_lr_h_sgd

    # --- Experimento LR-Híbrido 2: Adam -> Complex_SIGMA-D ---
    model_lr_h_sigmad = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam_d = optim.Adam(model_lr_h_sigmad.parameters(), lr=LR_ADAM)
    optimizer_lr_h_sigmad = Complex_SIGMA_D(
        model_lr_h_sigmad.parameters(), lr=LR_SIGMA, beta=0.9
    )

    history_lr_h_sigmad, time_lr_h_sigmad = run_experiment(
        experiment_name="[LR] Adam -> Complex SIGMA-D",
        model=model_lr_h_sigmad,
        optimizer_config=[
            (optimizer_lr_h_adam_d, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sigmad, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> Complex SIGMA-D'] = history_lr_h_sigmad
    times_lr['Adam -> Complex SIGMA-D'] = time_lr_h_sigmad

    # --- Experimento LR-Híbrido 3: Adam -> Complex_SIGMA-C ---
    model_lr_h_sigmac = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam_c = optim.Adam(model_lr_h_sigmac.parameters(), lr=LR_ADAM)
    optimizer_lr_h_sigmac = Complex_SIGMA_C(
        model_lr_h_sigmac.parameters(), lr=LR_SIGMA, beta=0.9
    )

    history_lr_h_sigmac, time_lr_h_sigmac = run_experiment(
        experiment_name="[LR] Adam -> Complex SIGMA-C",
        model=model_lr_h_sigmac,
        optimizer_config=[
            (optimizer_lr_h_adam_c, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sigmac, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> Complex SIGMA-C'] = history_lr_h_sigmac
    times_lr['Adam -> Complex SIGMA-C'] = time_lr_h_sigmac

    # ========================================================================
    # GERAÇÃO DE PLOTS E RESUMOS
    # ========================================================================

    # Gerar gráficos .pdf
    generate_nn_plots(results_nn, times_nn, N_EPOCHS_NN_PHASE1)
    generate_lr_plots(results_lr, times_lr, N_EPOCHS_LR_PHASE1)

    # Imprimir resumos estatísticos no console

    print("\n" + "="*80)
    print("RESUMO FINAL - REDES NEURAIS (COMPLEXAS)")
    print("="*80)
    print(f"\n{'Experimento':<30} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 70)
    for name in results_nn.keys():
        acc_final = results_nn[name]['test_acc'][-1]
        loss_final = results_nn[name]['test_loss'][-1]
        time_final = times_nn[name]
        print(f"{name:<30} | {acc_final:>9.2f}% | {loss_final:>10.4f} | {time_final:>9.2f}s")

    print("\n" + "="*80)
    print("RESUMO FINAL - REGRESSÃO LOGÍSTICA (COMPLEXA)")
    print("="*80)
    print(f"\n{'Otimizador':<30} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 70)
    for name in results_lr.keys():
        acc_final = results_lr[name]['test_acc'][-1]
        loss_final = results_lr[name]['train_loss'][-1]
        time_final = times_lr[name]
        print(f"{name:<30} | {acc_final:>9.2f}% | {loss_final:>10.6f} | {time_final:>9.2f}s")

    print("\n" + "="*80)
    print("Benchmark complexo concluído com sucesso!")
    print("="*80)


if __name__ == "__main__":
    main()
