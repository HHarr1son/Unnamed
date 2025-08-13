import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SubjectAwareAttention(nn.Module):
    """Subject-aware multi-head attention for IAM"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        subject_embed_dim: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Subject-specific bias projections
        self.subject_q_proj = nn.Linear(subject_embed_dim, embed_dim)
        self.subject_k_proj = nn.Linear(subject_embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        subject_embed: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input features
            subject_embed: [B, D_s] subject embeddings
            key_padding_mask: [B, L] mask for padding tokens
            
        Returns:
            torch.Tensor: [B, L, D] attended features
        """
        B, L, D = x.shape
        
        # Standard projections
        q = self.q_proj(x)  # [B, L, D]
        k = self.k_proj(x)  # [B, L, D]
        v = self.v_proj(x)  # [B, L, D]
        
        # Subject-aware bias
        subject_q_bias = self.subject_q_proj(subject_embed).unsqueeze(1)  # [B, 1, D]
        subject_k_bias = self.subject_k_proj(subject_embed).unsqueeze(1)  # [B, 1, D]
        
        # Add subject bias
        q = q + subject_q_bias
        k = k + subject_k_bias
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D_h]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D_h]
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D_h]
        
        # Attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)  # [B, H, L, D_h]
        out = out.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        
        return self.out_proj(out)


class CrossStreamAttention(nn.Module):
    """Cross-stream attention for DSGNet spatial-temporal fusion"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Cross-attention: spatial queries, temporal keys/values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            spatial_features: [B, L_s, D] spatial stream features
            temporal_features: [B, L_t, D] temporal stream features
            
        Returns:
            torch.Tensor: [B, L_s, D] cross-attended features
        """
        B, L_s, D = spatial_features.shape
        L_t = temporal_features.shape[1]
        
        # Queries from spatial, Keys/Values from temporal
        q = self.q_proj(spatial_features)  # [B, L_s, D]
        k = self.k_proj(temporal_features)  # [B, L_t, D]
        v = self.v_proj(temporal_features)  # [B, L_t, D]
        
        # Multi-head reshape
        q = q.view(B, L_s, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_s, D_h]
        k = k.view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_t, D_h]
        v = v.view(B, L_t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_t, D_h]
        
        # Cross-attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_s, L_t]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)  # [B, H, L_s, D_h]
        out = out.transpose(1, 2).contiguous().view(B, L_s, D)  # [B, L_s, D]
        
        return self.out_proj(out)


class SpatialAttention(nn.Module):
    """Spatial attention for electrode relationships"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, spatial_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, D] electrode features (C = channels, D = features)
            spatial_mask: [C, C] spatial adjacency mask
            
        Returns:
            torch.Tensor: [B, C, D] spatially attended features
        """
        residual = x
        attn_out, _ = self.attention(x, x, x, attn_mask=spatial_mask)
        return self.norm(residual + attn_out)


class TemporalAttention(nn.Module):
    """Temporal attention for time series modeling"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, max_seq_len: int = 500):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding for temporal modeling
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] temporal features
            
        Returns:
            torch.Tensor: [B, T, D] temporally attended features
        """
        B, T, D = x.shape
        
        # Add positional encoding
        pos_embed = self.pos_encoding[:T].unsqueeze(0).expand(B, -1, -1)
        x_pos = x + pos_embed
        
        residual = x_pos
        attn_out, _ = self.attention(x_pos, x_pos, x_pos)
        return self.norm(residual + attn_out)


class GatedAttention(nn.Module):
    """Gated attention for adaptive feature selection"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input features
            
        Returns:
            torch.Tensor: [B, L, D] gated attended features
        """
        # Standard attention
        attn_out, _ = self.attention(x, x, x)
        
        # Gating mechanism
        gate_weights = self.gate(x)
        
        return x + gate_weights * attn_out


class RotaryPositionalEmbedding(nn.Module):
    """RoPE positional embedding for better temporal modeling"""
    
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        
        # Compute rotation frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device):
        """Compute cos/sin values for given sequence length"""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding
        
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            torch.Tensor: [B, L, D] with rotary position encoding
        """
        seq_len = x.shape[1]
        cos, sin = self._compute_cos_sin(seq_len, x.device)
        
        # Split into pairs for rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # Apply rotation
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return x_rotated


def create_attention(
    attention_type: str,
    embed_dim: int,
    num_heads: int = 8,
    **kwargs
) -> nn.Module:
    """Factory function for attention mechanisms
    
    Args:
        attention_type: Type of attention mechanism
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        **kwargs: Additional parameters
        
    Returns:
        Attention module
    """
    if attention_type == "subject_aware":
        return SubjectAwareAttention(embed_dim, num_heads, **kwargs)
    elif attention_type == "cross_stream":
        return CrossStreamAttention(embed_dim, num_heads, **kwargs)
    elif attention_type == "spatial":
        return SpatialAttention(embed_dim, num_heads)
    elif attention_type == "temporal":
        return TemporalAttention(embed_dim, num_heads, **kwargs)
    elif attention_type == "gated":
        return GatedAttention(embed_dim)
    elif attention_type == "standard":
        return nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


__all__ = [
    "SubjectAwareAttention",
    "CrossStreamAttention", 
    "SpatialAttention",
    "TemporalAttention",
    "GatedAttention",
    "RotaryPositionalEmbedding",
    "create_attention"
]