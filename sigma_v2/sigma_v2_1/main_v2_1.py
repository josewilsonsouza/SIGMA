"""
Benchmark Comparativo de Híbridos Sequenciais v2
======================================================
Script Principal para Execução (Versão 2)

Este script importa:
- Otimizadores de 'optimizers_v2.py'
- Modelos de 'models.py'
- Funções de utilidade de 'utils_v2.py'
- Funções de plotagem de 'plotting_v2.py'

Todos os otimizadores agora usam weight_decay para uma comparação
mais justa (AdamW, SGDW, SIGMA-D_v2, SIGMA-C_v2).

*** NOVO: Adicionado Experimento Cíclico (A->C->A->C) ***
"""

"""
Benchmark Comparativo de Híbridos Sequenciais v2.1
==================================================
Script Principal para Execução (Versão 2.1)

Testa os novos modos de momentum (sigma, classic, nesterov)
para o otimizador SIGMA-C v2.1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Importações dos arquivos locais v2
from models import MNISTNet, LogisticRegression
from optimizers_v2_1 import SIGMA_C_v2, SIGMA_D_v2
from utils_v2_1 import get_data_loaders, run_experiment
from plotting_v2_1 import generate_nn_plots, generate_lr_plots

def main():
    """Executa todos os experimentos v2.1 e gera análises comparativas."""
    
    # Configuração
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")
    
    N_EPOCHS_NN_TOTAL = 20
    N_EPOCHS_NN_PHASE1 = 10
    N_EPOCHS_NN_PHASE2 = N_EPOCHS_NN_TOTAL - N_EPOCHS_NN_PHASE1
    
    N_EPOCHS_LR_TOTAL = 30
    N_EPOCHS_LR_PHASE1 = 15
    N_EPOCHS_LR_PHASE2 = N_EPOCHS_LR_TOTAL - N_EPOCHS_LR_PHASE1
    
    LR_ADAM = 0.001
    LR_SGD = 0.01
    LR_SIGMA = 0.01
    WD = 0
    
    train_loader, test_loader = get_data_loaders()
    loss_fn = nn.CrossEntropyLoss()
    
    # =======================================================================
    # PARTE 1: EXPERIMENTOS COM REDES NEURAIS
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 1: EXPERIMENTOS v2.1 COM REDES NEURAIS (MODOS DE MOMENTUM)")
    print("="*80)
    
    base_model = MNISTNet().to(DEVICE)
    
    results_nn = {}
    times_nn = {}
    
    # --- EXPERIMENTO 1: Adam (Baseline) ---
    model_adam = copy.deepcopy(base_model)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=LR_ADAM, weight_decay=WD)
    
    history_adam, time_adam = run_experiment(
        experiment_name="Adam (Baseline)",
        model=model_adam,
        optimizer_config=[(optimizer_adam, N_EPOCHS_NN_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam (Baseline)'] = history_adam
    times_nn['Adam (Baseline)'] = time_adam
    
    # --- EXPERIMENTO 2: Híbrido (Adam → SGD+M Clássico) ---
    model_hybrid_sgd = copy.deepcopy(base_model)
    optimizer_adam_phase1_sgd = optim.Adam(model_hybrid_sgd.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sgd_phase2 = optim.SGD(model_hybrid_sgd.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD, nesterov=False)
    
    history_hybrid_sgd, time_hybrid_sgd = run_experiment(
        experiment_name="Adam -> SGD+M (Classic)",
        model=model_hybrid_sgd,
        optimizer_config=[
            (optimizer_adam_phase1_sgd, N_EPOCHS_NN_PHASE1),
            (optimizer_sgd_phase2, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> SGD+M (Classic)'] = history_hybrid_sgd
    times_nn['Adam -> SGD+M (Classic)'] = time_hybrid_sgd

    # --- EXPERIMENTO 3: Híbrido (Adam → SGD+Nesterov) ---
    model_hybrid_sgd_n = copy.deepcopy(base_model)
    optimizer_adam_phase1_sgd_n = optim.Adam(model_hybrid_sgd_n.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sgd_phase2_n = optim.SGD(model_hybrid_sgd_n.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD, nesterov=True)
    
    history_hybrid_sgd_n, time_hybrid_sgd_n = run_experiment(
        experiment_name="Adam -> SGD+M (Nesterov)",
        model=model_hybrid_sgd_n,
        optimizer_config=[
            (optimizer_adam_phase1_sgd_n, N_EPOCHS_NN_PHASE1),
            (optimizer_sgd_phase2_n, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn['Adam -> SGD+M (Nesterov)'] = history_hybrid_sgd_n
    times_nn['Adam -> SGD+M (Nesterov)'] = time_hybrid_sgd_n

    # --- EXPERIMENTO 4: Híbrido (Adam → SIGMA-C 'sigma') ---
    model_hybrid_sigma_c_sig = copy.deepcopy(base_model)
    optimizer_adam_phase1_c_sig = optim.Adam(model_hybrid_sigma_c_sig.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sigma_c_sig = SIGMA_C_v2(
        model_hybrid_sigma_c_sig.parameters(),
        lr=LR_SIGMA,
        momentum_type='sigma',
        score_beta=0.9,
        weight_decay=WD
    )
    
    history_hybrid_sigma_c_sig, time_hybrid_sigma_c_sig = run_experiment(
        experiment_name="Adam -> SIGMA-C (Momentum='sigma')",
        model=model_hybrid_sigma_c_sig,
        optimizer_config=[
            (optimizer_adam_phase1_c_sig, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_c_sig, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn["Adam -> SIGMA-C (Momentum='sigma')"] = history_hybrid_sigma_c_sig
    times_nn["Adam -> SIGMA-C (Momentum='sigma')"] = time_hybrid_sigma_c_sig

    # --- EXPERIMENTO 5: Híbrido (Adam → SIGMA-C 'classic') ---
    model_hybrid_sigma_c_cls = copy.deepcopy(base_model)
    optimizer_adam_phase1_c_cls = optim.Adam(model_hybrid_sigma_c_cls.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sigma_c_cls = SIGMA_C_v2(
        model_hybrid_sigma_c_cls.parameters(),
        lr=LR_SIGMA,
        momentum_type='classic',
        momentum=0.9,
        weight_decay=WD
    )
    
    history_hybrid_sigma_c_cls, time_hybrid_sigma_c_cls = run_experiment(
        experiment_name="Adam -> SIGMA-C (Momentum='classic')",
        model=model_hybrid_sigma_c_cls,
        optimizer_config=[
            (optimizer_adam_phase1_c_cls, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_c_cls, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn["Adam -> SIGMA-C (Momentum='classic')"] = history_hybrid_sigma_c_cls
    times_nn["Adam -> SIGMA-C (Momentum='classic')"] = time_hybrid_sigma_c_cls

    # --- EXPERIMENTO 6: Híbrido (Adam → SIGMA-C 'nesterov') ---
    model_hybrid_sigma_c_nes = copy.deepcopy(base_model)
    optimizer_adam_phase1_c_nes = optim.Adam(model_hybrid_sigma_c_nes.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_sigma_c_nes = SIGMA_C_v2(
        model_hybrid_sigma_c_nes.parameters(),
        lr=LR_SIGMA,
        momentum_type='nesterov',
        momentum=0.9,
        weight_decay=WD
    )
    
    history_hybrid_sigma_c_nes, time_hybrid_sigma_c_nes = run_experiment(
        experiment_name="Adam -> SIGMA-C (Momentum='nesterov')",
        model=model_hybrid_sigma_c_nes,
        optimizer_config=[
            (optimizer_adam_phase1_c_nes, N_EPOCHS_NN_PHASE1),
            (optimizer_sigma_c_nes, N_EPOCHS_NN_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_NN_TOTAL
    )
    results_nn["Adam -> SIGMA-C (Momentum='nesterov')"] = history_hybrid_sigma_c_nes
    times_nn["Adam -> SIGMA-C (Momentum='nesterov')"] = time_hybrid_sigma_c_nes
    
    
    # =======================================================================
    # PARTE 2: EXPERIMENTOS COM REGRESSÃO LOGÍSTICA
    # (Não modificado, focado na Parte 1)
    # =======================================================================
    
    print("\n" + "="*80)
    print("PARTE 2: EXPERIMENTOS v2 COM REGRESSÃO LOGÍSTICA (PROBLEMA CONVEXO)")
    print("="*80)
    
    base_logistic = LogisticRegression().to(DEVICE)
    results_lr = {}
    times_lr = {}
    
    # --- Experimento LR-Puro 1: Adam ---
    model_lr_adam = copy.deepcopy(base_logistic)
    optimizer_lr_adam = optim.Adam(model_lr_adam.parameters(), lr=LR_ADAM, weight_decay=WD)
    history_lr_adam, time_lr_adam = run_experiment(
        experiment_name="[LR] Adam (Puro)",
        model=model_lr_adam,
        optimizer_config=[(optimizer_lr_adam, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam (Puro)'] = history_lr_adam
    times_lr['Adam (Puro)'] = time_lr_adam
    
    # --- Experimento LR-Puro 2: SGD+Momentum ---
    model_lr_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_sgd = optim.SGD(model_lr_sgd.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    history_lr_sgd, time_lr_sgd = run_experiment(
        experiment_name="[LR] SGD+M (Puro)",
        model=model_lr_sgd,
        optimizer_config=[(optimizer_lr_sgd, N_EPOCHS_LR_TOTAL)],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['SGD+M (Puro)'] = history_lr_sgd
    times_lr['SGD+M (Puro)'] = time_lr_sgd
    
    # --- Experimento LR-Híbrido 1: Adam -> SGD+M ---
    model_lr_h_sgd = copy.deepcopy(base_logistic)
    optimizer_lr_h_adam = optim.Adam(model_lr_h_sgd.parameters(), lr=LR_ADAM, weight_decay=WD)
    optimizer_lr_h_sgd = optim.SGD(model_lr_h_sgd.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=WD)
    
    history_lr_h_sgd, time_lr_h_sgd = run_experiment(
        experiment_name="[LR] Adam -> SGD+M",
        model=model_lr_h_sgd,
        optimizer_config=[
            (optimizer_lr_h_adam, N_EPOCHS_LR_PHASE1),
            (optimizer_lr_h_sgd, N_EPOCHS_LR_PHASE2)
        ],
        train_loader=train_loader, test_loader=test_loader,
        device=DEVICE, loss_fn=loss_fn, n_epochs=N_EPOCHS_LR_TOTAL
    )
    results_lr['Adam -> SGD+M'] = history_lr_h_sgd
    times_lr['Adam -> SGD+M'] = time_lr_h_sgd
    
    # ========================================================================
    # GERAÇÃO DE PLOTS E RESUMOS
    # ========================================================================
    
    generate_nn_plots(results_nn, times_nn, N_EPOCHS_NN_PHASE1)
    generate_lr_plots(results_lr, times_lr, N_EPOCHS_LR_PHASE1)
    
    # --- Resumo Final (Terminal) ---
    
    print("\n" + "="*80)
    print("RESUMO FINAL - REDES NEURAIS (v2.1 - Modos de Momentum)")
    print("="*80)
    
    print(f"\n{'Experimento':<35} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 75)
    
    for name in results_nn.keys():
        acc_final = results_nn[name]['test_acc'][-1]
        loss_final = results_nn[name]['test_loss'][-1]
        time_final = times_nn[name]
        print(f"{name:<35} | {acc_final:>9.2f}% | {loss_final:>10.4f} | {time_final:>9.2f}s")
    
    print("\n" + "="*80)
    print("RESUMO FINAL - REGRESSÃO LOGÍSTICA (v2.1)")
    print("="*80)
    
    print(f"\n{'Otimizador':<25} | {'Acc Final':<10} | {'Loss Final':<11} | {'Tempo (s)':<10}")
    print("-" * 70)
    
    for name in results_lr.keys():
        acc_final = results_lr[name]['test_acc'][-1]
        loss_final = results_lr[name]['train_loss'][-1]
        time_final = times_lr[name]
        print(f"{name:<25} | {acc_final:>9.2f}% | {loss_final:>10.6f} | {time_final:>9.2f}s")

    print("\n" + "="*80)
    print("Benchmark v2.1 concluído com sucesso!")
    print("="*80)


if __name__ == "__main__":
    main()