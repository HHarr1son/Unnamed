"""
Normalization Modules for NEED Framework

Various normalization techniques used throughout the NEED models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LayerNorm(nn.Module):
    """Standard Layer Normalization with optional bias"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (from LLaMA)"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class GroupNorm(nn.Module):
    """Group Normalization for channel-wise features"""
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Norm with condition-dependent scale and shift"""
    
    def __init__(self, normalized_shape: int, condition_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        
        # Condition-dependent parameters
        self.scale_proj = nn.Linear(condition_dim, normalized_shape)
        self.shift_proj = nn.Linear(condition_dim, normalized_shape)
        
        # Initialize to identity transformation: scale near 0 (so 1+scale=1), shift near 0
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Standard layer norm
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        # Condition-dependent scale and shift
        scale = 1.0 + self.scale_proj(condition)
        shift = self.shift_proj(condition)
        
        # Apply conditioning
        if condition.dim() == 2 and x.dim() == 3:
            scale = scale.unsqueeze(1)  # [B, 1, D]
            shift = shift.unsqueeze(1)  # [B, 1, D]
        
        return scale * normalized + shift


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.scale_proj = nn.Linear(condition_dim, feature_dim)
        self.shift_proj = nn.Linear(condition_dim, feature_dim)
        
        # Initialize to identity: scale near 0 (so 1+scale=1), shift near 0
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.scale_proj(condition)
        shift = self.shift_proj(condition)
        
        # Broadcast for different tensor shapes
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(-2)
            shift = shift.unsqueeze(-2)
        
        return scale * x + shift


class BatchNorm1d(nn.Module):
    """Batch Normalization for 1D features"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input shapes
        if x.dim() == 3:  # [B, L, D] -> [B, D, L]
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
            return x
        else:  # [B, D]
            return self.norm(x)


class ChannelNorm(nn.Module):
    """Channel-wise normalization for EEG signals"""
    
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] - normalize along time dimension for each channel
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        return self.weight * normalized + self.bias


class SpectralNorm(nn.Module):
    """Spectral normalization wrapper"""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        # Add spectral norm to the module
        self.module = nn.utils.spectral_norm(module, name, n_power_iterations)
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class InstanceNorm(nn.Module):
    """Instance normalization for individual samples"""
    
    def __init__(self, num_features: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, ...] - normalize across spatial dimensions for each sample and channel
        shape = x.shape
        x = x.view(shape[0], shape[1], -1)  # [B, C, spatial_dims]
        
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        if self.affine:
            weight = self.weight.view(1, -1, 1)
            bias = self.bias.view(1, -1, 1)
            normalized = weight * normalized + bias
        
        return normalized.view(shape)


class ConditionalBatchNorm(nn.Module):
    """Conditional Batch Normalization"""
    
    def __init__(self, num_features: int, condition_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        
        # Standard BN parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Condition-dependent parameters
        self.scale_proj = nn.Linear(condition_dim, num_features)
        self.shift_proj = nn.Linear(condition_dim, num_features)
        
        nn.init.ones_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean([0, 2])  # [C]
            var = x.var([0, 2], unbiased=False)
            
            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        normalized = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        normalized = self.weight.view(1, -1, 1) * normalized + self.bias.view(1, -1, 1)
        
        # Apply condition
        scale = 1.0 + self.scale_proj(condition).view(-1, self.num_features, 1)
        shift = self.shift_proj(condition).view(-1, self.num_features, 1)
        
        return scale * normalized + shift


def create_normalization(
    norm_type: str,
    dim: int,
    **kwargs
) -> nn.Module:
    """Factory function for normalization layers
    
    Args:
        norm_type: Type of normalization
        dim: Feature dimension
        **kwargs: Additional parameters
        
    Returns:
        Normalization module
    """
    if norm_type == "layer_norm":
        return LayerNorm(dim, **kwargs)
    elif norm_type == "rms_norm":
        return RMSNorm(dim, **kwargs)
    elif norm_type == "group_norm":
        # Expect num_groups in kwargs; map dim to num_channels
        if "num_groups" not in kwargs:
            raise ValueError("group_norm requires 'num_groups' in kwargs")
        return GroupNorm(num_groups=kwargs["num_groups"], num_channels=dim, eps=kwargs.get("eps", 1e-6))
    elif norm_type == "batch_norm":
        return BatchNorm1d(dim, **kwargs)
    elif norm_type == "channel_norm":
        return ChannelNorm(dim, **kwargs)
    elif norm_type == "instance_norm":
        return InstanceNorm(dim, **kwargs)
    elif norm_type == "adaptive_layer_norm":
        return AdaptiveLayerNorm(dim, **kwargs)
    elif norm_type == "film":
        return FiLM(dim, **kwargs)
    elif norm_type == "conditional_batch_norm":
        return ConditionalBatchNorm(dim, **kwargs)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


__all__ = [
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "AdaptiveLayerNorm",
    "FiLM",
    "BatchNorm1d",
    "ChannelNorm",
    "SpectralNorm",
    "InstanceNorm", 
    "ConditionalBatchNorm",
    "create_normalization"
]