import matplotlib.pyplot as plt


def generate_nn_plots(results_nn, times_nn, phase1_epochs):
    """
    Gera gráficos de comparação para experimentos com Redes Neurais Complexas.

    Args:
        results_nn: Dict com históricos de treinamento {nome: history}
        times_nn: Dict com tempos de execução {nome: tempo}
        phase1_epochs: Número de épocas da primeira fase (para linha de troca)
    """
    print("\nGerando gráficos de comparação (Redes Neurais Complexas)...")
    fig_nn, axes_nn = plt.subplots(2, 2, figsize=(18, 12))

    colors_nn = {
        'Complex Adam (Baseline)': '#1f77b4',
        'Adam -> Complex SGD+M': '#d62728',
        'Adam -> Complex SIGMA-D': '#2ca02c',
        'Adam -> Complex SIGMA-C': '#9467bd',
    }
    markers_nn = {
        'Complex Adam (Baseline)': 'o',
        'Adam -> Complex SGD+M': 'v',
        'Adam -> Complex SIGMA-D': '^',
        'Adam -> Complex SIGMA-C': 'P',
    }

    # Gráfico 1: Acurácia
    ax1 = axes_nn[0, 0]
    for name, history in results_nn.items():
        ax1.plot(history['test_acc'], label=name, color=colors_nn[name],
                marker=markers_nn[name], markersize=4, linewidth=2)
    ax1.axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Troca')
    ax1.set_title('Acurácia (Rede Neural Complexa)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Loss de Teste (escala log)
    ax2 = axes_nn[0, 1]
    for name, history in results_nn.items():
        ax2.plot(history['test_loss'], label=name, color=colors_nn[name],
                marker=markers_nn[name], markersize=4, linewidth=2)
    ax2.axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.7, label='Troca')
    ax2.set_title('Loss de Teste (Rede Neural Complexa)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Loss no Teste (escala log)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Gráfico 3: Loss de Treino
    ax3 = axes_nn[1, 0]
    for name, history in results_nn.items():
        ax3.plot(history['train_loss'], label=name, color=colors_nn[name],
                marker=markers_nn[name], markersize=3, linewidth=2, alpha=0.8)
    ax3.axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--',
               linewidth=1.5, alpha=0.7)
    ax3.set_title('Loss de Treino (Rede Neural Complexa)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Época', fontsize=12)
    ax3.set_ylabel('Loss de Treino', fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Gráfico 4: Eficiência Computacional
    ax4 = axes_nn[1, 1]
    names_nn = list(times_nn.keys())
    time_values_nn = list(times_nn.values())
    bars = ax4.barh(names_nn, time_values_nn, color=[colors_nn[n] for n in names_nn])
    for i, (bar, val) in enumerate(zip(bars, time_values_nn)):
        ax4.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    ax4.set_title('Eficiência Computacional (Rede Neural Complexa)',
                 fontsize=14, fontweight='bold')
    ax4.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4.grid(axis='x', alpha=0.3)

    fig_nn.tight_layout()
    fig_nn.savefig('complex_sigma_hybrid_comparison_nn.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'complex_sigma_hybrid_comparison_nn.pdf'")


def generate_lr_plots(results_lr, times_lr, phase1_epochs):
    """
    Gera gráficos de comparação para experimentos com Regressão Logística Complexa.

    Args:
        results_lr: Dict com históricos de treinamento {nome: history}
        times_lr: Dict com tempos de execução {nome: tempo}
        phase1_epochs: Número de épocas da primeira fase (para linha de troca)
    """
    print("\nGerando gráficos de comparação (Regressão Logística Complexa)...")
    fig_lr, axes_lr = plt.subplots(2, 2, figsize=(18, 12))

    colors_lr = {
        'Complex Adam (Puro)': '#1f77b4',
        'Complex SGD+M (Puro)': '#ff7f0e',
        'Complex SIGMA-D (Puro)': '#2ca02c',
        'Complex SIGMA-C (Puro)': '#9467bd',
        'Adam -> Complex SGD+M': '#d62728',
        'Adam -> Complex SIGMA-D': '#8c564b',
        'Adam -> Complex SIGMA-C': '#e377c2',
    }
    markers_lr = { k: 'o' if 'Puro' in k else '^' for k in colors_lr.keys() }
    linestyles_lr = { k: ':' if 'Puro' in k else '-' for k in colors_lr.keys() }

    # Gráfico 1: Acurácia
    ax1_lr = axes_lr[0, 0]
    for name, history in results_lr.items():
        ax1_lr.plot(history['test_acc'], label=name, color=colors_lr[name],
                   linestyle=linestyles_lr[name], marker=markers_lr[name],
                   markersize=4, linewidth=2)
    ax1_lr.axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Troca')
    ax1_lr.set_title('Acurácia (Regressão Logística Complexa)',
                    fontsize=14, fontweight='bold')
    ax1_lr.set_xlabel('Época', fontsize=12)
    ax1_lr.set_ylabel('Acurácia no Teste (%)', fontsize=12)
    ax1_lr.legend(fontsize=9, loc='lower right')
    ax1_lr.grid(True, alpha=0.3)

    # Gráfico 2: Loss de Teste (escala log)
    ax2_lr = axes_lr[0, 1]
    for name, history in results_lr.items():
        ax2_lr.plot(history['test_loss'], label=name, color=colors_lr[name],
                   linestyle=linestyles_lr[name], marker=markers_lr[name],
                   markersize=4, linewidth=2)
    ax2_lr.axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Troca')
    ax2_lr.set_title('Loss de Teste (Regressão Logística Complexa)',
                    fontsize=14, fontweight='bold')
    ax2_lr.set_xlabel('Época', fontsize=12)
    ax2_lr.set_ylabel('Loss no Teste (escala log)', fontsize=12)
    ax2_lr.set_yscale('log')
    ax2_lr.legend(fontsize=9, loc='upper right')
    ax2_lr.grid(True, alpha=0.3)

    # Gráfico 3: Loss de Treino (escala log)
    ax3_lr = axes_lr[1, 0]
    for name, history in results_lr.items():
        ax3_lr.plot(history['train_loss'], label=name, color=colors_lr[name],
                   linestyle=linestyles_lr[name], marker=markers_lr[name],
                   markersize=4, linewidth=2)
    ax3_lr.axvline(x=phase1_epochs - 0.5, color='gray', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Troca')
    ax3_lr.set_title('Loss de Treino (Regressão Logística Complexa)',
                    fontsize=14, fontweight='bold')
    ax3_lr.set_xlabel('Época', fontsize=12)
    ax3_lr.set_ylabel('Loss de Treino (escala log)', fontsize=12)
    ax3_lr.set_yscale('log')
    ax3_lr.legend(fontsize=9, loc='upper right')
    ax3_lr.grid(True, alpha=0.3)

    # Gráfico 4: Eficiência Computacional
    ax4_lr = axes_lr[1, 1]
    names_lr = list(times_lr.keys())
    time_values_lr = list(times_lr.values())
    bars = ax4_lr.barh(names_lr, time_values_lr, color=[colors_lr[n] for n in names_lr])
    for i, (bar, val) in enumerate(zip(bars, time_values_lr)):
        ax4_lr.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
    ax4_lr.set_title('Eficiência Computacional (Regressão Logística Complexa)',
                    fontsize=14, fontweight='bold')
    ax4_lr.set_xlabel('Tempo de Treinamento (s)', fontsize=12)
    ax4_lr.grid(axis='x', alpha=0.3)

    fig_lr.tight_layout()
    fig_lr.savefig('complex_sigma_full_comparison_logistic.pdf', dpi=150, bbox_inches='tight')
    print("Gráfico salvo: 'complex_sigma_full_comparison_logistic.pdf'")
