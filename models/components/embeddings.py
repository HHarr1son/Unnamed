import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SubjectEmbedding(nn.Module):
    """Learnable subject embeddings for IAM"""
    
    def __init__(self, num_subjects: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_subjects = num_subjects
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(num_subjects, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small random values
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, subject_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            subject_ids: [B] subject indices
            
        Returns:
            torch.Tensor: [B, D] subject embeddings
        """
        embeds = self.embedding(subject_ids)
        return self.dropout(embeds)


class TaskEmbedding(nn.Module):
    """Task type embeddings for unified inference"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Two tasks: video=0, image=1
        self.embedding = nn.Embedding(2, embed_dim)
        
        # Initialize with distinct patterns
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0] *= 1.5  # Video task
            self.embedding.weight[1] *= -1.0  # Image task (opposite pattern)
    
    def forward(self, task_type: str, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            task_type: "video" or "image"
            batch_size: Batch size
            device: Device for tensor
            
        Returns:
            torch.Tensor: [B, D] task embeddings
        """
        task_id = 0 if task_type == "video" else 1
        task_ids = torch.full((batch_size,), task_id, dtype=torch.long, device=device)
        return self.embedding(task_ids)


class DatasetEmbedding(nn.Module):
    """Dataset-specific embeddings for multi-dataset training"""
    
    def __init__(self, dataset_names: list, embed_dim: int):
        super().__init__()
        self.dataset_names = dataset_names
        self.name_to_id = {name: i for i, name in enumerate(dataset_names)}
        
        self.embedding = nn.Embedding(len(dataset_names), embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, dataset_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            dataset_name: Name of dataset
            batch_size: Batch size
            device: Device for tensor
            
        Returns:
            torch.Tensor: [B, D] dataset embeddings
        """
        dataset_id = self.name_to_id.get(dataset_name, 0)
        dataset_ids = torch.full((batch_size,), dataset_id, dtype=torch.long, device=device)
        return self.embedding(dataset_ids)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings"""
    
    def __init__(self, embed_dim: int, max_len: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Create sinusoidal position embeddings
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                           -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, embed_dim]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            torch.Tensor: [B, L, D] with added positional embeddings
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embeddings"""
    
    def __init__(self, max_len: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] input tensor
            
        Returns:
            torch.Tensor: [B, L, D] with added positional embeddings
        """
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.embedding(positions)
        return self.dropout(x + pos_embeds)


class ChannelEmbedding(nn.Module):
    """EEG channel/electrode embeddings"""
    
    def __init__(self, num_channels: int, embed_dim: int):
        super().__init__()
        self.num_channels = num_channels
        self.embedding = nn.Embedding(num_channels, embed_dim)
        
        # Initialize based on standard electrode positions if possible
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size
            device: Device for tensor
            
        Returns:
            torch.Tensor: [B, C, D] channel embeddings
        """
        channel_ids = torch.arange(self.num_channels, device=device)
        channel_embeds = self.embedding(channel_ids)  # [C, D]
        return channel_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C, D]


class FrequencyEmbedding(nn.Module):
    """Frequency band embeddings for spectral processing"""
    
    def __init__(self, frequency_bands: list, embed_dim: int):
        super().__init__()
        self.frequency_bands = frequency_bands  # [(low, high), ...]
        self.num_bands = len(frequency_bands)
        
        self.embedding = nn.Embedding(self.num_bands, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            batch_size: Batch size
            device: Device for tensor
            
        Returns:
            torch.Tensor: [B, F, D] frequency band embeddings
        """
        band_ids = torch.arange(self.num_bands, device=device)
        band_embeds = self.embedding(band_ids)  # [F, D]
        return band_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, F, D]


class SphericalHarmonicEmbedding(nn.Module):
    """Spherical harmonic embeddings for electrode positions"""
    
    def __init__(self, max_degree: int, embed_dim: int):
        super().__init__()
        self.max_degree = max_degree
        
        # Calculate number of spherical harmonic coefficients
        num_coeffs = (max_degree + 1) ** 2
        
        self.coeff_embedding = nn.Linear(num_coeffs, embed_dim)
        
        # Learnable coefficients for spherical harmonics
        self.coefficients = nn.Parameter(torch.randn(num_coeffs) * 0.02)
    
    def forward(self, electrode_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            electrode_positions: [C, 3] electrode xyz coordinates
            
        Returns:
            torch.Tensor: [C, D] spherical harmonic embeddings
        """
        # Compute spherical coordinates
        x, y, z = electrode_positions[:, 0], electrode_positions[:, 1], electrode_positions[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2) + 1e-8
        
        theta = torch.acos(z / r)  # Polar angle
        phi = torch.atan2(y, x)    # Azimuthal angle
        
        # Compute spherical harmonic basis (simplified)
        harmonics = []
        for l in range(self.max_degree + 1):
            for m in range(-l, l + 1):
                # Simplified spherical harmonic computation
                if m == 0:
                    ylm = torch.cos(l * theta)
                elif m > 0:
                    ylm = torch.cos(m * phi) * torch.sin(l * theta)
                else:
                    ylm = torch.sin(abs(m) * phi) * torch.sin(l * theta)
                harmonics.append(ylm)
        
        harmonic_features = torch.stack(harmonics, dim=1)  # [C, num_coeffs]
        
        # Weight by learnable coefficients
        weighted_features = harmonic_features * self.coefficients.unsqueeze(0)
        
        return self.coeff_embedding(weighted_features)


class TimeEmbedding(nn.Module):
    """Time embeddings for diffusion timesteps"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] diffusion timesteps
            
        Returns:
            torch.Tensor: [B, D] time embeddings
        """
        # Sinusoidal time embedding
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if self.embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        
        return self.mlp(emb)


def create_embedding(
    embedding_type: str,
    embed_dim: int,
    **kwargs
) -> nn.Module:
    """Factory function for embeddings
    
    Args:
        embedding_type: Type of embedding
        embed_dim: Embedding dimension
        **kwargs: Additional parameters
        
    Returns:
        Embedding module
    """
    if embedding_type == "subject":
        return SubjectEmbedding(embed_dim=embed_dim, **kwargs)
    elif embedding_type == "task":
        return TaskEmbedding(embed_dim)
    elif embedding_type == "dataset":
        return DatasetEmbedding(embed_dim=embed_dim, **kwargs)
    elif embedding_type == "positional":
        return PositionalEmbedding(embed_dim, **kwargs)
    elif embedding_type == "learnable_pos":
        return LearnablePositionalEmbedding(embed_dim=embed_dim, **kwargs)
    elif embedding_type == "channel":
        return ChannelEmbedding(embed_dim=embed_dim, **kwargs)
    elif embedding_type == "frequency":
        return FrequencyEmbedding(embed_dim=embed_dim, **kwargs)
    elif embedding_type == "spherical":
        return SphericalHarmonicEmbedding(embed_dim=embed_dim, **kwargs)
    elif embedding_type == "time":
        return TimeEmbedding(embed_dim)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


__all__ = [
    "SubjectEmbedding",
    "TaskEmbedding", 
    "DatasetEmbedding",
    "PositionalEmbedding",
    "LearnablePositionalEmbedding",
    "ChannelEmbedding",
    "FrequencyEmbedding",
    "SphericalHarmonicEmbedding",
    "TimeEmbedding",
    "create_embedding"
]