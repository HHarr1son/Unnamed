"""
Gradient Utilities for NEED Framework

Gradient manipulation tools for adversarial training and optimization.
"""

import torch
import torch.nn as nn
from typing import Iterable, Optional, Union


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float = 1.0) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_val * grad_output, None


class GradientScaling(torch.autograd.Function):
    """Scale gradients by a constant factor"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.scale * grad_output, None


class GradientClipping(torch.autograd.Function):
    """Clip gradients during backward pass"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, clip_value: float = 1.0) -> torch.Tensor:
        ctx.clip_value = clip_value
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return torch.clamp(grad_output, -ctx.clip_value, ctx.clip_value), None


class GradientReversal(nn.Module):
    """Gradient Reversal Module for domain adaptation"""
    
    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalLayer.apply(x, self.lambda_val)
    
    def set_lambda(self, lambda_val: float):
        """Update lambda value"""
        self.lambda_val = lambda_val


class AdaptiveGradientReversal(nn.Module):
    """Adaptive GRL with scheduling"""
    
    def __init__(
        self, 
        initial_lambda: float = 0.0,
        max_lambda: float = 1.0,
        schedule: str = "linear"  # "linear", "exponential"
    ):
        super().__init__()
        self.initial_lambda = initial_lambda
        self.max_lambda = max_lambda
        self.schedule = schedule
        self.current_lambda = initial_lambda
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalLayer.apply(x, self.current_lambda)
    
    def update_lambda(self, progress: float):
        """Update lambda based on training progress
        
        Args:
            progress: Training progress [0, 1]
        """
        if self.schedule == "linear":
            self.current_lambda = self.initial_lambda + progress * (self.max_lambda - self.initial_lambda)
        elif self.schedule == "exponential":
            alpha = 10.0
            factor = 2.0 / (1.0 + torch.exp(-alpha * progress)) - 1.0
            self.current_lambda = self.initial_lambda + factor * (self.max_lambda - self.initial_lambda)
        else:
            self.current_lambda = self.max_lambda


class GradientScaler(nn.Module):
    """Gradient scaling module"""
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientScaling.apply(x, self.scale)


def clip_grad_norm_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> torch.Tensor:
    """Clip gradient norm of parameters
    
    Args:
        parameters: Parameters to clip
        max_norm: Maximum norm
        norm_type: Type of norm (default: 2.0)
        error_if_nonfinite: Raise error for non-finite gradients
        
    Returns:
        Total norm of gradients
    """
    return torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, norm_type, error_if_nonfinite
    )


def clip_grad_value_(
    parameters: Iterable[torch.Tensor],
    clip_value: float
) -> None:
    """Clip gradient values of parameters
    
    Args:
        parameters: Parameters to clip
        clip_value: Maximum absolute gradient value
    """
    torch.nn.utils.clip_grad_value_(parameters, clip_value)


def get_grad_norm(
    parameters: Iterable[torch.Tensor],
    norm_type: float = 2.0
) -> torch.Tensor:
    """Get gradient norm without clipping
    
    Args:
        parameters: Parameters to compute norm for
        norm_type: Type of norm
        
    Returns:
        Gradient norm
    """
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.data, norm_type) for p in parameters]),
            norm_type
        )
    
    return total_norm


def zero_grad_if_needed(
    parameters: Iterable[torch.Tensor],
    condition: bool = True
) -> None:
    """Zero gradients conditionally
    
    Args:
        parameters: Parameters to zero
        condition: Whether to zero gradients
    """
    if condition:
        for p in parameters:
            if p.grad is not None:
                p.grad.zero_()


def accumulate_gradients(
    parameters: Iterable[torch.Tensor],
    scale: float = 1.0
) -> None:
    """Scale accumulated gradients
    
    Args:
        parameters: Parameters with gradients
        scale: Scale factor
    """
    for p in parameters:
        if p.grad is not None:
            p.grad.data *= scale


def check_grad_finite(parameters: Iterable[torch.Tensor]) -> bool:
    """Check if all gradients are finite
    
    Args:
        parameters: Parameters to check
        
    Returns:
        True if all gradients are finite
    """
    for p in parameters:
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                return False
    return True


class GradientMonitor:
    """Monitor gradient statistics during training"""
    
    def __init__(self):
        self.grad_norms = []
        self.grad_means = []
        self.grad_stds = []
    
    def update(self, parameters: Iterable[torch.Tensor]):
        """Update gradient statistics
        
        Args:
            parameters: Parameters to monitor
        """
        grad_norm = get_grad_norm(parameters).item()
        self.grad_norms.append(grad_norm)
        
        # Compute mean and std of all gradients
        all_grads = []
        for p in parameters:
            if p.grad is not None:
                all_grads.append(p.grad.data.flatten())
        
        if all_grads:
            all_grads = torch.cat(all_grads)
            self.grad_means.append(all_grads.mean().item())
            self.grad_stds.append(all_grads.std().item())
    
    def get_stats(self) -> dict:
        """Get gradient statistics"""
        return {
            "grad_norm_mean": sum(self.grad_norms) / len(self.grad_norms) if self.grad_norms else 0.0,
            "grad_norm_max": max(self.grad_norms) if self.grad_norms else 0.0,
            "grad_mean": sum(self.grad_means) / len(self.grad_means) if self.grad_means else 0.0,
            "grad_std": sum(self.grad_stds) / len(self.grad_stds) if self.grad_stds else 0.0,
        }
    
    def reset(self):
        """Reset statistics"""
        self.grad_norms.clear()
        self.grad_means.clear()
        self.grad_stds.clear()


def apply_gradient_penalty(
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    discriminator: nn.Module,
    lambda_gp: float = 10.0
) -> torch.Tensor:
    """Apply gradient penalty for WGAN-GP
    
    Args:
        real_data: Real data samples
        fake_data: Generated data samples  
        discriminator: Discriminator network
        lambda_gp: Gradient penalty weight
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, device=real_data.device)
    alpha = alpha.expand_as(real_data)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # Get discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty


__all__ = [
    "GradientReversalLayer",
    "GradientScaling", 
    "GradientClipping",
    "GradientReversal",
    "AdaptiveGradientReversal",
    "GradientScaler",
    "clip_grad_norm_",
    "clip_grad_value_",
    "get_grad_norm",
    "zero_grad_if_needed",
    "accumulate_gradients", 
    "check_grad_finite",
    "GradientMonitor",
    "apply_gradient_penalty"
]