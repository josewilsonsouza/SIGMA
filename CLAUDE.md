# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SIGMA (Score-Informed Geometric Momentum Adaptation) is a PyTorch-based optimization framework that implements novel stochastic optimization algorithms based on geometric convexity scores computed from the loss function.

The project compares SIGMA optimizers against standard baselines (Adam, SGD+Momentum) on MNIST classification tasks using both neural networks and logistic regression models.

## Repository Structure

The codebase contains three main implementations:

- **sigma_v1/**: Basic SIGMA implementation with two optimizer variants (SIGMA-D and SIGMA-C)
- **sigma_v2/**: Enhanced version with advanced features (warmup, weight decay, gradient clipping, AMSGrad, second-order approximation)
- **sigma_complex/**: Complex-valued neural network implementation of SIGMA optimizers

Each directory is self-contained with its own models, optimizers, utilities, and plotting modules.

## Core Architecture

### SIGMA Optimizers

The framework implements two score-based optimizers:

1. **SIGMA-D** (Theorem 1 - Point D): Computes a geometric score based on parameter interpolation at point D
   - Score formula uses `D1 = (¸_prev * ¸_t) / (¸_prev + ¸_t + µ)` and `D2 = (¸_t * L_prev + ¸_prev * L_t) / (¸_prev + ¸_t + µ)`
   - Requires passing `loss_item` to `step()` method

2. **SIGMA-C** (Theorem 2 - Point C): Computes a geometric score based on loss interpolation at point C
   - Score formula uses `C1 = (|L_prev| * ¸_t + |L_t| * ¸_prev) / (|L_prev| + |L_t| + µ)` and `C2 = (L_t * L_prev) / (|L_prev| + |L_t| + µ)`
   - Also requires passing `loss_item` to `step()` method

Both optimizers support:
- Optional momentum on the score via `beta` parameter
- Score clipping via `alpha_min` and `alpha_max` bounds

**CRITICAL**: SIGMA optimizers require the loss value to be passed to the step function:
```python
optimizer.step(loss_item=loss.item())
```

### Hybrid Sequential Training

The main experiment design uses a **hybrid sequential approach**:
1. **Phase 1**: Train with Adam for initial epochs to get to a good region
2. **Phase 2**: Switch to SIGMA (or SGD+M for control) for fine-tuning

This hybrid approach is the core experimental design comparing SIGMA's fine-tuning capabilities against SGD+Momentum.

### Model Architectures

- **MNISTNet**: 3-layer feedforward network (784 ’ 128 ’ 64 ’ 10) with ReLU activations
- **LogisticRegression**: Single linear layer (784 ’ 10)
- **ComplexMNISTNet**: Complex-valued network using CReLU activation (ReLU applied separately to real and imaginary parts)
- **ComplexLogisticRegression**: Complex-valued single layer

## Common Commands

### Running Experiments

```bash
# Run basic SIGMA experiments (v1)
cd sigma_v1
python main.py

# Run enhanced SIGMA experiments (v2 with advanced features)
cd sigma_v2
python main_v2.py

# Run complex-valued SIGMA experiments
cd sigma_complex
python sigma_complex.py
```

### Expected Outputs

All experiments:
- Print detailed training progress to console with loss and accuracy per epoch
- Generate PDF plots comparing optimizer performance
- Display final summary tables with accuracy, loss, and timing metrics

V1 generates:
- `sigma_hybrid_comparison_nn.pdf` (neural network results)
- `sigma_full_comparison_logistic.pdf` (logistic regression results)

V2 generates:
- `sigma_v2_hybrid_comparison_nn.pdf`
- `sigma_v2_full_comparison_logistic.pdf`

Complex generates:
- `complex_sigma_hybrid_comparison_nn.pdf`
- `complex_sigma_full_comparison_logistic.pdf`

### Dependencies

The project requires:
- PyTorch (with torchvision for MNIST datasets)
- matplotlib (for plotting)
- numpy

Install with:
```bash
pip install torch torchvision matplotlib numpy
```

## Key Implementation Details

### Data Loading
- MNIST dataset is automatically downloaded to `./data/` directory
- Default batch size: 128 for training, 1000 for testing
- Normalization: mean=0.1307, std=0.3081

### Training Configuration

**Neural Network Experiments:**
- Total epochs: 20 (10 Adam + 10 second optimizer)
- Adam learning rate: 0.001
- SGD/SIGMA learning rate: 0.01
- SIGMA beta (momentum): 0.9
- SIGMA alpha bounds: [0.1, 2.0]

**Logistic Regression Experiments:**
- Total epochs: 30 (15 + 15 split)
- Same learning rates and hyperparameters as neural network

**V2 Additions:**
- Weight decay: 0.01 (applied to all optimizers for fair comparison)
- Cyclic experiment: (Adam ’ SIGMA-C) × 2 with 5 epochs per phase

### Loss Function Behavior

SIGMA optimizers use the loss value to compute adaptive scores, making them distinct from gradient-based optimizers. The score Ã modulates the update: `¸_{t+1} = ¸_t - · · g_t ™ Ã_t`

### Complex-Valued Networks

The complex implementation:
- Converts real MNIST data to complex dtype (`torch.cfloat`)
- Uses magnitude of output for loss computation and predictions
- Applies CReLU activation: `CReLU(z) = ReLU(Re(z)) + i·ReLU(Im(z))`
- SIGMA-D produces complex scores; SIGMA-C produces real scores

## Module Organization

Each version follows the same structure:

- **main.py / main_v2.py / sigma_complex.py**: Entry point that orchestrates all experiments
- **models.py**: Neural network architectures
- **optimizers.py / optimizers_v2.py**: SIGMA optimizer implementations
- **utils.py / utils_v2.py**: Data loading and training loop functions
- **plotting.py / plotting_v2.py**: Visualization generation

The modular design allows easy experimentation with different optimizer variants and model architectures.

## License

Apache License 2.0 - Copyright 2025 José Wilson C. Souza
