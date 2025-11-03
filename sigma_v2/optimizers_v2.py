"""
Otimizadores SIGMA v2.0 - Implementações Avançadas

Contém:
- SIGMA_D_v2: Otimizador com score do Teorema 1 (Ponto D),
              baseado em sigma_v2.py.
- SIGMA_C_v2: Otimizador com score do Teorema 2 (Ponto C),
              adaptado com os mesmos recursos avançados.
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
    Versão 2.0 - Baseada no Teorema 1 (Ponto D)
    
    Possui recursos modernos: warmup, weight decay desacoplado, grad_clip,
    amsgrad, e aproximação de 2ª ordem.
    """

    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        beta: Optional[float] = None,
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
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay inválido: {weight_decay}")
        if grad_clip is not None and not grad_clip > 0:
            raise ValueError(f"Gradient clipping inválido: {grad_clip}")
        if not warmup_steps >= 0:
            raise ValueError(f"Warmup steps inválido: {warmup_steps}")
            
        defaults = dict(
            lr=lr,
            beta=beta,
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
        
        # Estado global do otimizador
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
            raise ValueError(
                "SIGMA.step() requer o argumento 'loss_item'. "
                "Exemplo: optimizer.step(loss_item=loss.item())"
            )
        
        # Atualizar estado global
        loss_t = float(loss_item)
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        step = self.state['global_step']
        
        for group in self.param_groups:
            base_lr = group['lr']
            lr = self._get_lr_with_warmup(base_lr)
            beta = group['beta']
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
                    if beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data)
                    if second_order:
                        state['grad_prev'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_score'] = torch.ones_like(p.data)
                
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
                
                # --- Aplicação de Momentum (SIGMA-M) ---
                if beta is not None:
                    score_mom = state['score_momentum']
                    score_mom.mul_(beta).add_(sigma_raw, alpha=1 - beta)
                    sigma_to_use = score_mom
                else:
                    sigma_to_use = sigma_raw
                
                # --- AMSGrad Variant (opcional) ---
                if amsgrad:
                    max_score = state['max_score']
                    torch.max(max_score, torch.abs(sigma_to_use), out=max_score)
                    sigma_to_use = max_score
                
                sigma_to_use = torch.clamp(sigma_to_use, alpha_min, alpha_max)

                # --- Atualização dos Parâmetros ---
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                p.data.addcmul_(grad, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
                if second_order:
                    state['grad_prev'] = grad.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        return f"SIGMA-D_v2(lr={self.defaults['lr']}, beta={self.defaults['beta']})"

# ============================================================================
# OTIMIZADOR 2: SIGMA_C_v2 (Score Teorema 2 - Ponto C)
# ============================================================================

class SIGMA_C_v2(Optimizer):
    """
    Implementa o otimizador SIGMA (Score-Informed Geometric Momentum Adaptation).
    Versão 2.0 - Baseada no Teorema 2 (Ponto C)
    
    Possui os mesmos recursos avançados do SIGMA_D_v2 (warmup, weight decay,
    grad_clip, amsgrad, 2ª ordem), mas usa a formulação de 
    score do Teorema 2.
    """

    def __init__(
        self, 
        params, 
        lr: float = 1e-3,
        beta: Optional[float] = None,
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
        if beta is not None and not 0.0 <= beta < 1.0:
            raise ValueError(f"Beta inválido: {beta}. Deve estar em [0, 1)")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon inválido: {eps}")
        if not 0.0 < alpha_min < alpha_max:
            raise ValueError(f"Limites alpha inválidos: [{alpha_min}, {alpha_max}]")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay inválido: {weight_decay}")
        if grad_clip is not None and not grad_clip > 0:
            raise ValueError(f"Gradient clipping inválido: {grad_clip}")
        if not warmup_steps >= 0:
            raise ValueError(f"Warmup steps inválido: {warmup_steps}")
            
        defaults = dict(
            lr=lr,
            beta=beta,
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
        
        # Estado global do otimizador
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
            raise ValueError(
                "SIGMA.step() requer o argumento 'loss_item'. "
                "Exemplo: optimizer.step(loss_item=loss.item())"
            )
        
        # Atualizar estado global
        loss_t = float(loss_item)
        self.state['global_step'] += 1
        loss_prev = self.state['global_loss_prev']
        step = self.state['global_step']
        
        for group in self.param_groups:
            base_lr = group['lr']
            lr = self._get_lr_with_warmup(base_lr)
            beta = group['beta']
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
                    if beta is not None:
                        state['score_momentum'] = torch.ones_like(p.data)
                    if second_order:
                        state['grad_prev'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['max_score'] = torch.ones_like(p.data)
                
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
                
                # --- AMSGrad Variant (opcional) ---
                if amsgrad:
                    max_score = state['max_score']
                    torch.max(max_score, torch.abs(sigma_to_use), out=max_score)
                    sigma_to_use = max_score
                
                sigma_to_use = torch.clamp(sigma_to_use, alpha_min, alpha_max)

                # --- Atualização dos Parâmetros ---
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                
                p.data.addcmul_(grad, sigma_to_use, value=-lr)
                
                state['param_prev'] = theta_t.clone()
                if second_order:
                    state['grad_prev'] = grad.clone()
        
        self.state['global_loss_prev'] = loss_t
        return loss_t
    
    def __repr__(self):
        return f"SIGMA-C_v2(lr={self.defaults['lr']}, beta={self.defaults['beta']})"