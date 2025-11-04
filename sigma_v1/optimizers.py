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

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable
import math

# ============================================================================
# OTIMIZADOR 1: SIGMA-D (Score Teorema 1 - Ponto D) (baseado em sigma.py)
# ============================================================================

class SIGMA_D(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    Baseado no Teorema 1 (Ponto D).
    """
    def __init__(self, params, lr=1e-2, beta=None, alpha_min=0.1, alpha_max=2.0, eps=1e-8):
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
        super(SIGMA_D, self).__init__(params, defaults)
        
        # Estado global do otimizador
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item=None):
        if loss_item is None:
            raise ValueError(
                "SIGMA.step() requer o argumento 'loss_item'. "
                "Exemplo: optimizer.step(loss_item=loss.item())"
            )
            
        loss_t = loss_item
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                g_t = p.grad.data
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data)
                
                theta_prev = state['param_prev']
                
                # --- Cálculo do Score σ (Teorema 1 - Ponto D) ---
                if self.state['global_step'] > 1:
                    denom_xy = theta_prev + theta_t + eps
                    D1 = (theta_prev * theta_t) / denom_xy
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_xy
                    
                    # Aproximação de Taylor para f(2D₁)
                    f_proxy = loss_t + g_t * (2 * D1 - theta_t)
                    
                    sigma_raw = D2 / (f_proxy + eps)
                    
                    sigma_raw[torch.isnan(sigma_raw)] = 1.0
                    sigma_raw[torch.isinf(sigma_raw)] = 1.0
                else:
                    sigma_raw = torch.ones_like(p.data)
                
                # --- Aplicação de Momentum (SIGMA-M) ---
                if beta is not None:
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                # Clipping do score final
                sigma_to_use.clamp_(alpha_min, alpha_max)

                # --- Atualização dos Parâmetros ---
                p.data.addcmul_(g_t, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        variant = "SIGMA-M (Score D)" if self.defaults['beta'] is not None else "SIGMA-T (Score D)"
        return f"{variant}(lr={self.defaults['lr']}, beta={self.defaults['beta']})"

# ============================================================================
# OTIMIZADOR 2: SIGMA-C (Score Teorema 2 - Ponto C)
# ============================================================================

class SIGMA_C(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    Baseado no Teorema 2 (Ponto C) das suas anotações.
    """
    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        beta: Optional[float] = None,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        eps: float = 1e-8,
    ):
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
            eps=eps,
        )
        super(SIGMA_C, self).__init__(params, defaults)
        
        # Estado global do otimizador
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    @torch.no_grad()
    def step(self, loss_item: Optional[float] = None, closure: Optional[Callable] = None):
        if loss_item is None:
            raise ValueError(
                "SIGMA.step() requer o argumento 'loss_item'. "
                "Exemplo: optimizer.step(loss_item=loss.item())"
            )
        
        loss_t = float(loss_item)
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        step = self.state['global_step']
        
        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data)
                
                theta_prev = state['param_prev']
                
                # --- Cálculo do Score σ (Teorema 2 - Ponto C) ---
                if step > 1:
                    denom_L = abs(loss_prev) + abs(loss_t) + eps
                    C1 = (abs(loss_prev) * theta_t + abs(loss_t) * theta_prev) / denom_L
                    two_C2 = (2 * loss_t * loss_prev) / denom_L
                    
                    # Aproximação de Taylor de primeira ordem para f(C₁)
                    f_proxy = loss_t + grad * (C1 - theta_t)
                    f_proxy = torch.clamp(f_proxy, min=eps)
                    
                    sigma_raw = two_C2 / (f_proxy + eps)
                    
                    sigma_raw = torch.where(
                        torch.isnan(sigma_raw) | torch.isinf(sigma_raw),
                        torch.ones_like(sigma_raw),
                        sigma_raw
                    )
                else:
                    sigma_raw = torch.ones_like(p.data)
                
                # --- Aplicação de Momentum (SIGMA-M) ---
                if beta is not None:
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                # Clipping do score final
                sigma_to_use = torch.clamp(sigma_to_use, alpha_min, alpha_max)

                # --- Atualização dos Parâmetros ---
                p.data.addcmul_(grad, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        variant = "SIGMA-M (Score C)" if self.defaults['beta'] is not None else "SIGMA-T (Score C)"
        return f"{variant}(lr={self.defaults['lr']}, beta={self.defaults['beta']})"