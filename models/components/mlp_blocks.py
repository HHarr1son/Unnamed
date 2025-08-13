"""
MLP Blocks for NEED Framework

Various MLP architectures used throughout the NEED models.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union

from .activations import get_activation


class MLP(nn.Module):
    """Basic MLP with configurable layers and activations"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, List[int]],
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        bias: bool = True,
        batch_norm: bool = False
    ):
        super().__init__()
        
        # Convert single hidden dim to list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(get_activation(activation, hidden_dim if activation in ["swiglu", "geglu", "glu"] else None))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim, bias=bias))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualMLP(nn.Module):
    """MLP with residual connections"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            get_activation(activation, hidden_dim if activation in ["swiglu", "geglu", "glu"] else None),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class GatedMLP(nn.Module):
    """MLP with gating mechanism"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, output_dim)
        
        self.activation = get_activation(activation)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.activation(self.up_proj(x))
        return self.down_proj(gate * up)


class ExpertMLP(nn.Module):
    """Single expert for Mixture of Experts"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "swiglu"
    ):
        super().__init__()
        
        if activation == "swiglu":
            # SwiGLU expert (LLaMA style)
            self.w1 = nn.Linear(input_dim, hidden_dim, bias=False)  # gate
            self.w2 = nn.Linear(hidden_dim, output_dim, bias=False)  # down
            self.w3 = nn.Linear(input_dim, hidden_dim, bias=False)   # up
        else:
            # Standard expert
            self.w1 = nn.Linear(input_dim, hidden_dim)
            self.w2 = nn.Linear(hidden_dim, output_dim)
            self.activation = get_activation(activation)
        
        self.activation_type = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "swiglu":
            return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))
        else:
            return self.w2(self.activation(self.w1(x)))


class MixtureOfExperts(nn.Module):
    """Mixture of Experts MLP"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = "swiglu"
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router/gating network
        self.router = nn.Linear(input_dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, hidden_dim, output_dim, activation)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)  # [B*L, D]
        
        # Router scores
        router_logits = self.router(x_flat)  # [B*L, num_experts]
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Expert outputs
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts])  # [num_experts, B*L, output_dim]
        
        # Weighted combination
        output = torch.zeros(batch_size * seq_len, expert_outputs.shape[-1], device=x.device)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_probs[:, i].unsqueeze(-1)
            expert_out = expert_outputs[expert_idx, torch.arange(batch_size * seq_len)]
            output += expert_weight * expert_out
        
        return output.view(batch_size, seq_len, -1)


class AdaptiveMLP(nn.Module):
    """MLP with adaptive capacity based on input"""
    
    def __init__(
        self,
        input_dim: int,
        max_hidden_dim: int,
        output_dim: int,
        num_layers: int = 2
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Capacity predictor
        self.capacity_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers),
            nn.Sigmoid()
        )
        
        # Variable capacity layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for _ in range(num_layers):
            self.layers.append(nn.Linear(prev_dim, max_hidden_dim))
            prev_dim = max_hidden_dim
        
        self.output_layer = nn.Linear(max_hidden_dim, output_dim)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict capacity for each layer
        capacity_weights = self.capacity_predictor(x.mean(dim=1))  # [B, num_layers]
        
        h = x
        for i, layer in enumerate(self.layers):
            h_full = self.activation(layer(h))
            
            # Apply capacity weighting
            capacity = capacity_weights[:, i].unsqueeze(1).unsqueeze(1)  # [B, 1, 1]
            h = capacity * h_full + (1 - capacity) * h
        
        return self.output_layer(h)


class ProjectionMLP(nn.Module):
    """Simple projection MLP for feature alignment"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        activation: str = "gelu",
        final_activation: bool = False
    ):
        super().__init__()
        
        if num_layers == 1:
            self.layers = nn.Linear(input_dim, output_dim)
        else:
            hidden_dim = hidden_dim or (input_dim + output_dim) // 2
            
            layers = [nn.Linear(input_dim, hidden_dim)]
            
            for _ in range(num_layers - 2):
                layers.extend([
                    get_activation(activation),
                    nn.Linear(hidden_dim, hidden_dim)
                ])
            
            layers.extend([
                get_activation(activation),
                nn.Linear(hidden_dim, output_dim)
            ])
            
            if final_activation:
                layers.append(get_activation(activation))
            
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FusionMLP(nn.Module):
    """MLP for multi-modal feature fusion"""
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int,
        output_dim: int,
        fusion_method: str = "concat",  # "concat", "add", "attention"
        activation: str = "gelu"
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.input_dims = input_dims
        
        if fusion_method == "concat":
            total_input_dim = sum(input_dims)
        elif fusion_method == "add":
            assert all(dim == input_dims[0] for dim in input_dims), "All dims must be equal for add fusion"
            total_input_dim = input_dims[0]
        elif fusion_method == "attention":
            # Project all to same dimension first
            self.projections = nn.ModuleList([
                nn.Linear(dim, hidden_dim) for dim in input_dims
            ])
            self.attention = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
            total_input_dim = hidden_dim
        
        self.fusion_mlp = MLP(
            total_input_dim, hidden_dim, output_dim, 
            activation=activation, dropout=0.1
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if self.fusion_method == "concat":
            fused = torch.cat(features, dim=-1)
        elif self.fusion_method == "add":
            fused = torch.stack(features).sum(dim=0)
        elif self.fusion_method == "attention":
            # Project to same dimension
            projected = [proj(feat) for proj, feat in zip(self.projections, features)]
            stacked = torch.stack(projected, dim=1)  # [B, num_modalities, D]
            
            # Self-attention across modalities
            attended, _ = self.attention(stacked, stacked, stacked)
            fused = attended.mean(dim=1)  # Average across modalities
        
        return self.fusion_mlp(fused)


def create_mlp(
    mlp_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """Factory function for MLP blocks
    
    Args:
        mlp_type: Type of MLP
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional parameters
        
    Returns:
        MLP module
    """
    if mlp_type == "basic":
        return MLP(input_dim, output_dim=output_dim, **kwargs)
    elif mlp_type == "residual":
        return ResidualMLP(input_dim, **kwargs)
    elif mlp_type == "gated":
        return GatedMLP(input_dim, output_dim=output_dim, **kwargs)
    elif mlp_type == "moe":
        return MixtureOfExperts(input_dim, output_dim=output_dim, **kwargs)
    elif mlp_type == "adaptive":
        return AdaptiveMLP(input_dim, output_dim=output_dim, **kwargs)
    elif mlp_type == "projection":
        return ProjectionMLP(input_dim, output_dim, **kwargs)
    elif mlp_type == "fusion":
        return FusionMLP(output_dim=output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown MLP type: {mlp_type}")


__all__ = [
    "MLP",
    "ResidualMLP",
    "GatedMLP", 
    "ExpertMLP",
    "MixtureOfExperts",
    "AdaptiveMLP",
    "ProjectionMLP",
    "FusionMLP",
    "create_mlp"
]