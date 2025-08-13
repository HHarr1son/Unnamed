import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    """SwiGLU activation from PaLM/LLaMA"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate(x)) * self.up(x)


class GeGLU(nn.Module):
    """GeGLU activation (GELU-based gating)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
        self.up = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.gate(x)) * self.up(x)


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class Snake(nn.Module):
    """Snake activation with learnable frequency"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x).pow(2)


class QuickGELU(nn.Module):
    """Faster GELU approximation from CLIP"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class GLU(nn.Module):
    """Simple GLU (Gated Linear Unit)"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


def get_activation(name: str, dim: Optional[int] = None) -> nn.Module:
    """Factory function for activation functions
    
    Args:
        name: Activation name
        dim: Dimension for gated activations
        
    Returns:
        Activation module
    """
    name = name.lower()
    
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    elif name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(0.02, inplace=True)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "mish":
        return Mish()
    elif name == "snake":
        return Snake()
    elif name == "quick_gelu":
        return QuickGELU()
    elif name == "swiglu":
        if dim is None:
            raise ValueError("SwiGLU requires dim parameter")
        return SwiGLU(dim)
    elif name == "geglu":
        if dim is None:
            raise ValueError("GeGLU requires dim parameter")
        return GeGLU(dim)
    elif name == "glu":
        if dim is None:
            raise ValueError("GLU requires dim parameter")
        return GLU(dim)
    else:
        raise ValueError(f"Unknown activation: {name}")


__all__ = [
    "SwiGLU", "GeGLU", "Mish", "Snake", "QuickGELU", "GLU", "get_activation"
]