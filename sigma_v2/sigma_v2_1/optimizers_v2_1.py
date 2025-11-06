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
Otimizadores SIGMA v2.1 - Implementações Avançadas com Modos de Momentum

Contém:
- SIGMA_D_v2: Otimizador com score do Teorema 1 (Ponto D).
- SIGMA_C_v2: Otimizador com score do Teorema 2 (Ponto C).

ATUALIZAÇÃO (v2.1):
- Adicionado 'momentum_type' para escolher entre:
  1. 'sigma': Momentum no score σ (comportamento original)
  2. 'classic': Momentum clássico (Polyak) no gradiente corrigido
  3. 'nesterov': Momentum Nesterov no gradiente corrigido
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, Callable
import math

# ============================================================================
# OTIMIZADOR 1: SIGMA_D_v2 (Score Teorema 1 - Ponto D)
# ============================================================================

class SIGMA_D_v2(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    Versão 2.1 - Baseada no Teorema 1 (Ponto D)
    
    Recursos: warmup, weight decay, grad_clip, amsgrad, 2ª ordem
    Modos de Momentum: 'sigma', 'classic', 'nesterov'
    """

    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        score_beta: Optional[float] = 0.9, # Renomeado de 'beta'
        momentum_type: str = 'sigma',
        momentum: float = 0.9,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
        warmup_steps: int = 0,
        second_order: bool = False,
        amsgrad: bool = False
    ):
        # Validação de parâmetros
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if score_beta is not None and not 0.0 <= score_beta < 1.0:
            raise ValueError(f"Score Beta inválido: {score_beta}. Deve estar em [0, 1)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum do gradiente inválido: {momentum}")
        if momentum_type not in ['sigma', 'classic', 'nesterov', None]:
            raise ValueError(f"Momentum type inválido: {momentum_type}")
        
        defaults = dict(
            lr=lr,
            score_beta=score_beta,
            momentum_type=momentum_type,
            momentum=momentum,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            warmup_steps=warmup_steps,
            second_order=second_order,
            amsgrad=amsgrad
        )
        super(SIGMA_D_v2, self).__init__(params, defaults)
        
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    def _get_lr_with_warmup(self, base_lr: float) -> float:
        """Aplica warmup linear na taxa de aprendizado."""
        step = self.state['global_step']
        warmup = self.defaults['warmup_steps']
        
        if warmup == 0 or step >= warmup:
            return base_lr
        else:
            return base_lr * (step + 1) / warmup

    @torch.no_grad()
    def step(self, loss_item: Optional[float] = None, closure: Optional[Callable] = None):
        if loss_item is None:
            raise ValueError("SIGMA.step() requer 'loss_item'")
        
        loss_t = float(loss_item)
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        step = self.state['global_step']
        
        for group in self.param_groups:
            lr = self._get_lr_with_warmup(group['lr'])
            score_beta = group['score_beta']
            mu = group['momentum']
            momentum_type = group['momentum_type']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']
            weight_decay = group['weight_decay']
            grad_clip = group['grad_clip']
            second_order = group['second_order']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if grad_clip is not None:
                    grad.clamp_(-grad_clip, grad_clip)
                
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if second_order:
                        state['grad_prev'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_score'] = torch.ones_like(p.data)
                    
                    # *** Lógica de Momentum ***
                    if momentum_type == 'sigma' and score_beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data)
                    if momentum_type in ['classic', 'nesterov']:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                
                theta_prev = state['param_prev']
                
                # --- Cálculo do Score σ (Teorema 1 - Ponto D) ---
                if step > 1:
                    denom_xy = torch.abs(theta_prev) + torch.abs(theta_t) + eps
                    D1 = (theta_prev * theta_t) / denom_xy
                    D2 = (theta_t * loss_prev + theta_prev * loss_t) / denom_xy
                    
                    if second_order and 'grad_prev' in state:
                        grad_prev = state['grad_prev']
                        delta_theta = theta_t - theta_prev
                        delta_grad = grad - grad_prev
                        hess_approx = delta_grad / (delta_theta + eps)
                        diff = 2 * D1 - theta_t
                        f_proxy = loss_t + (grad * diff).sum().item()
                        f_proxy += 0.5 * (diff * hess_approx * diff).sum().item()
                        f_proxy = max(f_proxy, eps)
                    else:
                        f_proxy = loss_t + grad * (2 * D1 - theta_t)
                        f_proxy = torch.clamp(f_proxy, min=eps)
                    
                    sigma_raw = D2 / (f_proxy + eps)
                    
                    sigma_raw = torch.where(
                        torch.isnan(sigma_raw) | torch.isinf(sigma_raw),
                        torch.ones_like(sigma_raw),
                        sigma_raw
                    )
                else:
                    sigma_raw = torch.ones_like(p.data)
                
                # --- Aplicação de Momentum no Score (se aplicável) ---
                if momentum_type == 'sigma' and score_beta is not None:
                    score_mom = state['score_momentum']
                    score_mom.mul_(score_beta).add_(sigma_raw, alpha=1 - score_beta)
                    sigma_to_use = score_mom
                else:
                    # Usa o score instantâneo se o momentum for no gradiente
                    sigma_to_use = sigma_raw
                
                if amsgrad:
                    max_score = state['max_score']
                    torch.max(max_score, torch.abs(sigma_to_use), out=max_score)
                    sigma_to_use = max_score
                
                sigma_to_use = torch.clamp(sigma_to_use, alpha_min, alpha_max)

                # --- Cálculo do Gradiente Corrigido ---
                grad_corrigido = grad * sigma_to_use
                
                # --- Aplicação de Momentum no Gradiente (se aplicável) ---
                if momentum_type in ['classic', 'nesterov']:
                    buf = state['momentum_buffer']
                    # v_t+1 = mu * v_t + g_corrigido
                    buf.mul_(mu).add_(grad_corrigido)
                    
                    if momentum_type == 'nesterov':
                        # g_corrigido + mu * v_t+1
                        update_vec = grad_corrigido.add(buf, alpha=mu)
                    else:
                        # 'classic': usa v_t+1
                        update_vec = buf
                else:
                    # 'sigma' ou None: usa g_corrigido
                    update_vec = grad_corrigido

                # --- Atualização dos Parâmetros ---
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                # p.data = p.data - lr * update_vec
                p.data.add_(update_vec, alpha=-lr)
                
                # Salvar estado para próxima iteração
                state['param_prev'] = theta_t.clone()
                if second_order:
                    state['grad_prev'] = grad.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        return f"SIGMA-D_v2(lr={self.defaults['lr']}, momentum_type={self.defaults['momentum_type']})"

# ============================================================================
# OTIMIZADOR 2: SIGMA_C_v2 (Score Teorema 2 - Ponto C)
# ============================================================================

class SIGMA_C_v2(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    Versão 2.1 - Baseada no Teorema 2 (Ponto C)
    
    Recursos: warmup, weight decay, grad_clip, amsgrad, 2ª ordem
    Modos de Momentum: 'sigma', 'classic', 'nesterov'
    """

    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        score_beta: Optional[float] = 0.9, # Renomeado de 'beta'
        momentum_type: str = 'sigma',
        momentum: float = 0.9,
        alpha_min: float = 0.1,
        alpha_max: float = 2.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
        warmup_steps: int = 0,
        second_order: bool = False,
        amsgrad: bool = False
    ):
        # Validação de parâmetros (idêntica ao v2)
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate inválido: {lr}")
        if score_beta is not None and not 0.0 <= score_beta < 1.0:
            raise ValueError(f"Score Beta inválido: {score_beta}. Deve estar em [0, 1)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Momentum do gradiente inválido: {momentum}")
        if momentum_type not in ['sigma', 'classic', 'nesterov', None]:
            raise ValueError(f"Momentum type inválido: {momentum_type}")
            
        defaults = dict(
            lr=lr,
            score_beta=score_beta,
            momentum_type=momentum_type,
            momentum=momentum,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            eps=eps,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            warmup_steps=warmup_steps,
            second_order=second_order,
            amsgrad=amsgrad
        )
        super(SIGMA_C_v2, self).__init__(params, defaults)
        
        self.state['global_loss_prev'] = 0.0
        self.state['global_step'] = 0

    def _get_lr_with_warmup(self, base_lr: float) -> float:
        """Aplica warmup linear na taxa de aprendizado."""
        step = self.state['global_step']
        warmup = self.defaults['warmup_steps']
        
        if warmup == 0 or step >= warmup:
            return base_lr
        else:
            return base_lr * (step + 1) / warmup

    @torch.no_grad()
    def step(self, loss_item: Optional[float] = None, closure: Optional[Callable] = None):
        if loss_item is None:
            raise ValueError("SIGMA.step() requer 'loss_item'")
        
        loss_t = float(loss_item)
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        step = self.state['global_step']
        
        for group in self.param_groups:
            lr = self._get_lr_with_warmup(group['lr'])
            score_beta = group['score_beta']
            mu = group['momentum']
            momentum_type = group['momentum_type']
            alpha_min = group['alpha_min']
            alpha_max = group['alpha_max']
            eps = group['eps']
            weight_decay = group['weight_decay']
            grad_clip = group['grad_clip']
            second_order = group['second_order']
            amsgrad = group['amsgrad']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if grad_clip is not None:
                    grad.clamp_(-grad_clip, grad_clip)
                
                theta_t = p.data
                state = self.state[p]

                # --- Inicialização do Estado ---
                if len(state) == 0:
                    state['param_prev'] = theta_t.clone()
                    if second_order:
                        state['grad_prev'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_score'] = torch.ones_like(p.data)

                    # *** Lógica de Momentum ***
                    if momentum_type == 'sigma' and score_beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data.real) # Score C é real
                    if momentum_type in ['classic', 'nesterov']:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                
                theta_prev = state['param_prev']
                
                # --- Cálculo do Score σ (Teorema 2 - Ponto C) ---
                if step > 1:
                    denom_L = abs(loss_prev) + abs(loss_t) + eps
                    C1 = (abs(loss_prev) * theta_t + abs(loss_t) * theta_prev) / denom_L
                    two_C2 = (2 * loss_t * loss_prev) / denom_L
                    
                    if second_order and 'grad_prev' in state:
                        grad_prev = state['grad_prev']
                        delta_theta = theta_t - theta_prev
                        delta_grad = grad - grad_prev
                        hess_approx = delta_grad / (delta_theta + eps)
                        
                        diff = C1 - theta_t
                        f_proxy = loss_t + (grad * diff).sum().item()
                        f_proxy += 0.5 * (diff * hess_approx * diff).sum().item()
                        f_proxy = max(f_proxy, eps)
                    else:
                        f_proxy = loss_t + grad * (C1 - theta_t)
                        f_proxy = torch.clamp(f_proxy, min=eps)
                    
                    sigma_raw = two_C2 / (f_proxy + eps) # Score é REAL
                    
                    sigma_raw = torch.where(
                        torch.isnan(sigma_raw) | torch.isinf(sigma_raw),
                        torch.ones_like(sigma_raw),
                        sigma_raw
                    )
                else:
                    sigma_raw = torch.ones_like(p.data.real)
                
                # --- Aplicação de Momentum no Score (se aplicável) ---
                if momentum_type == 'sigma' and score_beta is not None:
                    score_mom = state['score_momentum'] # REAL
                    score_mom.mul_(score_beta).add_(sigma_raw, alpha=1 - score_beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                if amsgrad:
                    max_score = state['max_score'] # REAL
                    torch.max(max_score, torch.abs(sigma_to_use), out=max_score)
                    sigma_to_use = max_score
                
                sigma_to_use = torch.clamp(sigma_to_use, alpha_min, alpha_max)

                # --- Cálculo do Gradiente Corrigido ---
                grad_corrigido = grad * sigma_to_use # (Complexo * Real)
                
                # --- Aplicação de Momentum no Gradiente (se aplicável) ---
                if momentum_type in ['classic', 'nesterov']:
                    buf = state['momentum_buffer'] # COMPLEXO
                    # v_t+1 = mu * v_t + g_corrigido
                    buf.mul_(mu).add_(grad_corrigido)
                    
                    if momentum_type == 'nesterov':
                        # g_corrigido + mu * v_t+1
                        update_vec = grad_corrigido.add(buf, alpha=mu)
                    else:
                        # 'classic': usa v_t+1
                        update_vec = buf
                else:
                    # 'sigma' ou None: usa g_corrigido
                    update_vec = grad_corrigido
                
                # --- Atualização dos Parâmetros ---
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                p.data.add_(update_vec, alpha=-lr)
                
                state['param_prev'] = theta_t.clone()
                if second_order:
                    state['grad_prev'] = grad.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        return f"SIGMA-C_v2(lr={self.defaults['lr']}, momentum_type={self.defaults['momentum_type']})"