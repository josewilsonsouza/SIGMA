"""
Copyright 2025 José Wilson C. Souza

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Otimizadores SIGMA para Parâmetros Complexos

Implementações de SIGMA-D e SIGMA-C adaptadas para trabalhar com
redes neurais complexas (torch.cfloat).
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional

# ============================================================================
# OTIMIZADOR 1: Complex_SIGMA-D (Score Teorema 1 - Ponto D)
# ============================================================================

class Complex_SIGMA_D(Optimizer):
    """
    Implementação do SIGMA-D (Teorema 1) para PARÂMETROS COMPLEXOS.
    A loss (L_t, L_prev) é REAL.
    Os parâmetros (theta_t, theta_prev) e gradientes (g_t) são COMPLEXOS.
    O score (sigma) resultante é COMPLEXO.
    """
    def __init__(self, params, lr=1e-2, beta=None, alpha_min=0.1, alpha_max=2.0, eps=1e-8):
        # Definições idênticas à V1
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")

        defaults = dict(
            lr=lr,
            beta=beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps
        )
        super(Complex_SIGMA_D, self).__init__(params, defaults)

        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item=None):
        if loss_item is None:
            raise ValueError("SIGMA.step() requer 'loss_item'")

        loss_t = loss_item # REAL
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev'] # REAL

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g_t = p.grad.data # COMPLEXO
                theta_t = p.data # COMPLEXO
                state = self.state[p]

                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        # .clone() preserva o dtype complexo
                        state['score_momentum'] = torch.ones_like(p.data)

                theta_prev = state['param_prev'] # COMPLEXO

                if self.state['global_step'] > 1:
                    # O denominador é complexo
                    denom_xy = theta_prev + theta_t + eps

                    # D1 = (complex * complex) / complex -> COMPLEXO
                    D1 = (theta_prev * theta_t) / denom_xy
                    # D2 = (complex * real + complex * real) / complex -> COMPLEXO
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_xy

                    # Aproximação de Taylor para f(2D₁)
                    # diff = 2*D1 - theta_t -> COMPLEXO
                    diff = (2 * D1 - theta_t)
                    # g_t * diff -> COMPLEXO (produto element-wise)
                    # f_proxy = real + real(complex * complex) -> REAL
                    f_proxy = loss_t + (g_t * diff).real
                    f_proxy = torch.clamp(f_proxy, min=eps) # Clamping REAL

                    # sigma_raw = D2 (COMPLEXO) / f_proxy (REAL) -> COMPLEXO
                    sigma_raw = D2 / (f_proxy + eps)

                    # Tratamento de valores inválidos (aplicado a real e imag)
                    sigma_raw[torch.isnan(sigma_raw)] = 1.0 + 0.0j
                    sigma_raw[torch.isinf(sigma_raw)] = 1.0 + 0.0j
                else:
                    sigma_raw = torch.ones_like(p.data) # COMPLEXO (1.0 + 0.0j)

                if beta is not None:
                    score_mom = state['score_momentum'] # COMPLEXO
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw

                # Clipping da MAGNITUDE do score complexo
                sigma_abs = sigma_to_use.abs()
                sigma_abs.clamp_(alpha_min, alpha_max)
                # Reaplicar a fase original
                sigma_to_use = torch.polar(sigma_abs, sigma_to_use.angle())

                # Atualização: p.data = p.data - lr * (g_t * sigma_to_use)
                # Esta é uma multiplicação COMPLEXA
                p.data.addcmul_(g_t, sigma_to_use, value=-lr)

                state['param_prev'] = theta_t.clone()

        self.state['global_loss_prev'] = loss_t
        return loss_t

    def __repr__(self):
        variant = "Complex_SIGMA-M (Score D)" if self.defaults['beta'] is not None else "Complex_SIGMA-T (Score D)"
        return f"{variant}(lr={self.defaults['lr']}, beta={self.defaults['beta']})"


# ============================================================================
# OTIMIZADOR 2: Complex_SIGMA-C (Score Teorema 2 - Ponto C)
# ============================================================================

class Complex_SIGMA_C(Optimizer):
    """
    Implementação do SIGMA-C (Teorema 2) para PARÂMETROS COMPLEXOS.
    A loss (L_t, L_prev) é REAL.
    Os parâmetros (theta_t, theta_prev) e gradientes (g_t) são COMPLEXOS.
    O score (sigma) resultante é REAL.
    """
    def __init__(self, params, lr=1e-2, beta=None, alpha_min=0.1, alpha_max=2.0, eps=1e-8):
        # Definições idênticas à V1
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")

        defaults = dict(
            lr=lr,
            beta=beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps
        )
        super(Complex_SIGMA_C, self).__init__(params, defaults)

        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item=None):
        if loss_item is None:
            raise ValueError("SIGMA.step() requer 'loss_item'")

        loss_t = loss_item # REAL
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev'] # REAL

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                g_t = p.grad.data # COMPLEXO
                theta_t = p.data # COMPLEXO
                state = self.state[p]

                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        # Score momentum é REAL para SIGMA-C
                        state['score_momentum'] = torch.ones_like(p.data.real)

                theta_prev = state['param_prev'] # COMPLEXO

                if self.state['global_step'] > 1:
                    # Denominador é REAL
                    denom_L = abs(loss_prev) + abs(loss_t) + eps

                    # C1 = (real * complex + real * complex) / real -> COMPLEXO
                    C1 = (abs(loss_prev) * theta_t + abs(loss_t) * theta_prev) / denom_L

                    # C2 = (real * real) / real -> REAL
                    C2 = (loss_t * loss_prev) / denom_L

                    # Aproximação de Taylor para f(C₁)
                    # diff = C1 - theta_t -> COMPLEXO
                    diff = C1 - theta_t
                    # f_proxy = real + real(complex * complex) -> REAL
                    f_proxy = loss_t + (g_t * diff).real
                    f_proxy = torch.clamp(f_proxy, min=eps) # Clamping REAL

                    # sigma_raw = 2 * C2 (REAL) / f_proxy (REAL) -> REAL
                    sigma_raw = (2 * C2) / (f_proxy + eps)

                    sigma_raw[torch.isnan(sigma_raw)] = 1.0
                    sigma_raw[torch.isinf(sigma_raw)] = 1.0
                else:
                    sigma_raw = torch.ones_like(p.data.real) # REAL

                if beta is not None:
                    score_mom = state['score_momentum'] # REAL
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw

                sigma_to_use.clamp_(alpha_min, alpha_max) # REAL

                # Atualização: p.data = p.data - lr * (g_t * sigma_to_use)
                # Esta é uma multiplicação COMPLEXO * REAL
                p.data.addcmul_(g_t, sigma_to_use, value=-lr)

                state['param_prev'] = theta_t.clone()

        self.state['global_loss_prev'] = loss_t
        return loss_t

    def __repr__(self):
        variant = "Complex_SIGMA-M (Score C)" if self.defaults['beta'] is not None else "Complex_SIGMA-T (Score C)"
        return f"{variant}(lr={self.defaults['lr']}, beta={self.defaults['beta']})"
