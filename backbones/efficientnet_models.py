import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from torchvision import models
    from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B7_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class EfficientNetWrapper(nn.Module):
    """EfficientNet backbone wrapper for feature extraction"""
    
    def __init__(
        self,
        model_name: str = "tf_efficientnet_b4",
        pretrained: bool = True,
        device: str = "cuda",
        freeze: bool = True,
        num_classes: int = 0,
        drop_rate: float = 0.0
    ):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available. Install with: pip install timm")
        
        self.device = device
        self.model_name = model_name
        
        # Load model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            features_only=True  # Extract features from all blocks
        )
        
        # Get feature information
        self.feature_info = self.model.feature_info
        self.stage_dims = [info['num_chs'] for info in self.feature_info]
        self.stage_reductions = [info['reduction'] for info in self.feature_info]
        
        # Final feature dimension
        self.feature_dim = self.stage_dims[-1]
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through EfficientNet
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] input images
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] features
        """
        original_shape = x.shape
        if len(original_shape) == 5:  # Video frames [B, T, C, H, W]
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        # Extract features from all stages
        stage_features = self.model(x)
        
        # Use final stage features with global average pooling
        final_features = stage_features[-1]  # [B*T, C, H, W]
        features = nn.functional.adaptive_avg_pool2d(final_features, (1, 1))
        features = features.flatten(1)  # [B*T, D]
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
        
        return features
    
    def extract_multiscale_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from all EfficientNet stages
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            Dict with features from each stage
        """
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        # Get all stage features
        stage_features = self.model(x)
        
        features = {}
        for i, feat in enumerate(stage_features):
            # Global average pooling for feature vectors
            pooled = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            features[f'stage_{i}'] = pooled
            features[f'stage_{i}_spatial'] = feat
            
            # Add reduction factor info
            features[f'stage_{i}_reduction'] = self.stage_reductions[i]
        
        # Reshape for video if needed
        if len(original_shape) == 5:
            for key in features:
                if 'spatial' in key:
                    _, C, H, W = features[key].shape
                    features[key] = features[key].view(B, T, C, H, W)
                elif key.startswith('stage_') and not key.endswith('_reduction'):
                    features[key] = features[key].view(B, T, -1)
        
        return features
    
    def get_stage_features(self, x: torch.Tensor, stage_idx: int = -1) -> torch.Tensor:
        """Get spatial features from specific stage
        
        Args:
            x: [B, C, H, W] input images
            stage_idx: Which stage to extract from (-1 for last stage)
            
        Returns:
            torch.Tensor: [B, C, H, W] spatial features
        """
        stage_features = self.model(x)
        
        if stage_idx == -1:
            stage_idx = len(stage_features) - 1
        
        return stage_features[stage_idx]


class TorchVisionEfficientNetWrapper(nn.Module):
    """TorchVision EfficientNet wrapper (limited variants)"""
    
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        pretrained: bool = True,
        device: str = "cuda",
        freeze: bool = True,
        num_classes: Optional[int] = None
    ):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available")
        
        self.device = device
        self.model_name = model_name
        
        # Load model based on name
        if model_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b7":
            weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_b7(weights=weights)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Get feature dimension
        self.feature_dim = self.model.classifier[1].in_features
        
        # Remove classifier for feature extraction
        self.model.classifier = nn.Identity()
        
        # Add custom classifier if specified
        if num_classes is not None:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = None
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
        if self.classifier is not None:
            self.classifier.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TorchVision EfficientNet"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        features = self.model(x)
        
        if self.classifier is not None:
            features = self.classifier(features)
        
        if len(original_shape) == 5:
            features = features.view(B, T, -1)
        
        return features


class EfficientNetFeaturePyramid(nn.Module):
    """EfficientNet with Feature Pyramid Network"""
    
    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b4",
        device: str = "cuda",
        fpn_dim: int = 256
    ):
        super().__init__()
        
        # Load EfficientNet backbone
        self.backbone = EfficientNetWrapper(
            backbone_name, pretrained=True, device=device, freeze=True
        )
        
        # Get stage dimensions
        stage_dims = self.backbone.stage_dims
        
        # FPN components
        self.fpn_dim = fpn_dim
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, fpn_dim, 1) for dim in stage_dims
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) 
            for _ in stage_dims
        ])
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with FPN features"""
        # Extract backbone features
        backbone_features = self.backbone.extract_multiscale_features(x)
        
        # Get spatial features only
        spatial_features = [
            feat for key, feat in backbone_features.items() 
            if 'spatial' in key
        ]
        
        # Build FPN top-down
        fpn_features = {}
        prev_features = None
        
        for i in range(len(spatial_features) - 1, -1, -1):
            lateral = self.lateral_convs[i](spatial_features[i])
            
            if prev_features is not None:
                # Upsample and add
                upsampled = nn.functional.interpolate(
                    prev_features, size=lateral.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
                lateral += upsampled
            
            fpn_features[f'p{i+2}'] = self.fpn_convs[i](lateral)
            prev_features = fpn_features[f'p{i+2}']
        
        return fpn_features


class EfficientNetWithAttention(nn.Module):
    """EfficientNet with self-attention mechanism"""
    
    def __init__(
        self,
        backbone_name: str = "tf_efficientnet_b4",
        device: str = "cuda",
        attention_dim: int = 256,
        num_heads: int = 8
    ):
        super().__init__()
        
        # Load backbone
        self.backbone = EfficientNetWrapper(
            backbone_name, pretrained=True, device=device, freeze=True
        )
        
        # Attention mechanism
        self.attention_dim = attention_dim
        self.feature_projection = nn.Linear(self.backbone.feature_dim, attention_dim)
        
        self.self_attention = nn.MultiheadAttention(
            attention_dim, num_heads, batch_first=True
        )
        
        self.norm = nn.LayerNorm(attention_dim)
        self.dropout = nn.Dropout(0.1)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with attention over spatial features"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        # Get spatial features from last stage
        stage_features = self.backbone.model(x)
        spatial_features = stage_features[-1]  # [B, C, H, W]
        
        # Reshape for attention: [B, H*W, C]
        B, C, H, W = spatial_features.shape
        spatial_features = spatial_features.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        
        # Project to attention dimension
        features = self.feature_projection(spatial_features)  # [B, HW, attention_dim]
        
        # Self-attention
        attended, _ = self.self_attention(features, features, features)
        attended = self.norm(attended + features)  # Residual connection
        attended = self.dropout(attended)
        
        # Global average pooling
        features = attended.mean(dim=1)  # [B, attention_dim]
        
        if len(original_shape) == 5:
            features = features.view(B // T, T, -1)
        
        return features


def create_efficientnet_model(
    model_name: str = "tf_efficientnet_b4",
    pretrained: bool = True,
    device: str = "cuda",
    model_type: str = "standard",
    use_torchvision: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function for EfficientNet models
    
    Args:
        model_name: EfficientNet model variant
        pretrained: Use pretrained weights
        device: Device to load model on
        model_type: "standard", "fpn", "attention"
        use_torchvision: Use TorchVision implementation (limited variants)
        
    Returns:
        EfficientNet model wrapper
    """
    
    if model_type == "fpn":
        return EfficientNetFeaturePyramid(model_name, device, **kwargs)
    
    elif model_type == "attention":
        return EfficientNetWithAttention(model_name, device, **kwargs)
    
    elif use_torchvision:
        # Convert TIMM names to TorchVision format
        tv_name_map = {
            "tf_efficientnet_b0": "efficientnet_b0",
            "tf_efficientnet_b7": "efficientnet_b7"
        }
        tv_name = tv_name_map.get(model_name, "efficientnet_b0")
        return TorchVisionEfficientNetWrapper(tv_name, pretrained, device, **kwargs)
    
    else:
        return EfficientNetWrapper(model_name, pretrained, device, **kwargs)


# Common EfficientNet variants
EFFICIENTNET_VARIANTS = {
    "efficientnet_b0": "tf_efficientnet_b0",
    "efficientnet_b1": "tf_efficientnet_b1", 
    "efficientnet_b2": "tf_efficientnet_b2",
    "efficientnet_b3": "tf_efficientnet_b3",
    "efficientnet_b4": "tf_efficientnet_b4",
    "efficientnet_b5": "tf_efficientnet_b5",
    "efficientnet_b6": "tf_efficientnet_b6",
    "efficientnet_b7": "tf_efficientnet_b7",
    "efficientnet_v2_s": "tf_efficientnetv2_s",
    "efficientnet_v2_m": "tf_efficientnetv2_m",
    "efficientnet_v2_l": "tf_efficientnetv2_l"
}


__all__ = [
    "EfficientNetWrapper",
    "TorchVisionEfficientNetWrapper",
    "EfficientNetFeaturePyramid",
    "EfficientNetWithAttention",
    "create_efficientnet_model",
    "EFFICIENTNET_VARIANTS"
]