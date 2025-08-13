import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import ConvNextModel, AutoImageProcessor, ConvNextConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ConvNeXtWrapper(nn.Module):
    """ConvNeXt backbone wrapper for feature extraction"""
    
    def __init__(
        self,
        model_name: str = "facebook/convnext-base-224-22k",
        device: str = "cuda",
        freeze: bool = True,
        use_pooler: bool = True
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        self.device = device
        self.model_name = model_name
        self.use_pooler = use_pooler
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ConvNextModel.from_pretrained(model_name)
        
        # Get model configuration
        self.config = self.model.config
        self.hidden_sizes = self.config.hidden_sizes
        self.num_stages = len(self.hidden_sizes)
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvNeXt
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] input images
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] features
        """
        original_shape = x.shape
        if len(original_shape) == 5:  # Video frames [B, T, C, H, W]
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        # Forward through model
        outputs = self.model(pixel_values=x)
        
        if self.use_pooler and hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output  # [B*T, D]
        else:
            # Global average pooling over the last hidden state
            last_hidden = outputs.last_hidden_state  # [B*T, H, W, C]
            # ConvNeXt uses channels-last format, convert to channels-first
            last_hidden = last_hidden.permute(0, 3, 1, 2)  # [B*T, C, H, W]
            features = nn.functional.adaptive_avg_pool2d(last_hidden, (1, 1))
            features = features.flatten(1)  # [B*T, D]
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
        
        return features
    
    def extract_hierarchical_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from all ConvNeXt stages
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            Dict with features from each stage
        """
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        # Get all hidden states
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        features = {}
        
        # Process each stage
        for i, hidden_state in enumerate(hidden_states):
            # ConvNeXt hidden states are in [B, H, W, C] format
            stage_features = hidden_state.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            # Global average pooling for feature vector
            pooled_features = nn.functional.adaptive_avg_pool2d(stage_features, (1, 1))
            pooled_features = pooled_features.flatten(1)  # [B, C]
            
            features[f'stage_{i}'] = pooled_features
            features[f'stage_{i}_spatial'] = stage_features
        
        # Reshape for video if needed
        if len(original_shape) == 5:
            for key in features:
                if 'spatial' in key:
                    _, C, H, W = features[key].shape
                    features[key] = features[key].view(B, T, C, H, W)
                else:
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
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Get features from specified stage
        if stage_idx == -1:
            stage_idx = len(hidden_states) - 1
        
        stage_features = hidden_states[stage_idx]  # [B, H, W, C]
        stage_features = stage_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return stage_features


class TIMMConvNeXtWrapper(nn.Module):
    """TIMM ConvNeXt wrapper with more variants"""
    
    def __init__(
        self,
        model_name: str = "convnext_base",
        pretrained: bool = True,
        device: str = "cuda",
        freeze: bool = True,
        num_classes: int = 0
    ):
        super().__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available. Install with: pip install timm")
        
        self.device = device
        self.model_name = model_name
        
        # Load model with feature extraction
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            features_only=True
        )
        
        # Get feature information
        self.feature_info = self.model.feature_info
        self.stage_dims = [info['num_chs'] for info in self.feature_info]
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning final stage features"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        # Get all stage features
        stage_features = self.model(x)
        
        # Use final stage and apply global average pooling
        final_features = stage_features[-1]  # [B, C, H, W]
        features = nn.functional.adaptive_avg_pool2d(final_features, (1, 1))
        features = features.flatten(1)  # [B, C]
        
        if len(original_shape) == 5:
            features = features.view(B, T, -1)
        
        return features
    
    def extract_all_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from all stages"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        stage_features = self.model(x)
        
        features = {}
        for i, feat in enumerate(stage_features):
            # Global pooling for classification features
            pooled = nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
            features[f'stage_{i}'] = pooled
            features[f'stage_{i}_spatial'] = feat
        
        if len(original_shape) == 5:
            for key in features:
                if 'spatial' in key:
                    _, C, H, W = features[key].shape
                    features[key] = features[key].view(B, T, C, H, W)
                else:
                    features[key] = features[key].view(B, T, -1)
        
        return features


class ConvNeXtWithLinearProbe(nn.Module):
    """ConvNeXt with linear probing head for downstream tasks"""
    
    def __init__(
        self,
        backbone_name: str = "facebook/convnext-base-224-22k",
        num_classes: int = 1000,
        device: str = "cuda",
        freeze_backbone: bool = True,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Load frozen backbone
        self.backbone = ConvNeXtWrapper(
            backbone_name, device=device, freeze=freeze_backbone
        )
        
        # Get feature dimension
        feature_dim = self.backbone.hidden_sizes[-1]
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with classification head"""
        features = self.backbone(x)
        return self.classifier(features)


class ConvNeXtFeaturePyramid(nn.Module):
    """ConvNeXt with Feature Pyramid Network"""
    
    def __init__(
        self,
        backbone_name: str = "convnext_base",
        device: str = "cuda",
        use_timm: bool = True
    ):
        super().__init__()
        
        # Load ConvNeXt backbone
        if use_timm:
            self.backbone = TIMMConvNeXtWrapper(
                backbone_name, pretrained=True, device=device, freeze=True
            )
            stage_dims = self.backbone.stage_dims
        else:
            self.backbone = ConvNeXtWrapper(
                f"facebook/{backbone_name.replace('_', '-')}-224-22k",
                device=device, freeze=True
            )
            stage_dims = self.backbone.hidden_sizes
        
        # FPN components
        self.fpn_dim = 256
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, self.fpn_dim, 1) for dim in stage_dims
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(self.fpn_dim, self.fpn_dim, 3, padding=1) 
            for _ in stage_dims
        ])
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with FPN features"""
        # Extract backbone features
        if hasattr(self.backbone, 'extract_all_stages'):
            backbone_features = self.backbone.extract_all_stages(x)
        else:
            backbone_features = self.backbone.extract_hierarchical_features(x)
        
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
                lateral += nn.functional.interpolate(
                    prev_features, scale_factor=2, mode='bilinear', align_corners=False
                )
            
            fpn_features[f'p{i+2}'] = self.fpn_convs[i](lateral)
            prev_features = fpn_features[f'p{i+2}']
        
        return fpn_features


def create_convnext_model(
    model_name: str = "convnext_base",
    pretrained: bool = True,
    device: str = "cuda",
    use_timm: bool = False,
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """Factory function for ConvNeXt models
    
    Args:
        model_name: ConvNeXt model variant
        pretrained: Use pretrained weights
        device: Device to load model on
        use_timm: Use TIMM implementation
        model_type: "standard", "probe", "fpn"
        
    Returns:
        ConvNeXt model wrapper
    """
    
    if model_type == "fpn":
        return ConvNeXtFeaturePyramid(model_name, device, use_timm, **kwargs)
    
    elif model_type == "probe":
        if use_timm:
            # Convert to HF format for linear probe
            hf_name_map = {
                "convnext_tiny": "facebook/convnext-tiny-224",
                "convnext_small": "facebook/convnext-small-224", 
                "convnext_base": "facebook/convnext-base-224-22k",
                "convnext_large": "facebook/convnext-large-224-22k",
                "convnext_xlarge": "facebook/convnext-xlarge-224-22k"
            }
            hf_name = hf_name_map.get(model_name, "facebook/convnext-base-224-22k")
        else:
            hf_name = model_name
        return ConvNeXtWithLinearProbe(hf_name, device=device, **kwargs)
    
    elif use_timm:
        return TIMMConvNeXtWrapper(model_name, pretrained, device, **kwargs)
    
    else:
        # Convert TIMM names to HuggingFace format
        hf_name_map = {
            "convnext_tiny": "facebook/convnext-tiny-224",
            "convnext_small": "facebook/convnext-small-224",
            "convnext_base": "facebook/convnext-base-224-22k",
            "convnext_large": "facebook/convnext-large-224-22k",
            "convnext_xlarge": "facebook/convnext-xlarge-224-22k"
        }
        hf_name = hf_name_map.get(model_name, model_name)
        return ConvNeXtWrapper(hf_name, device, **kwargs)


__all__ = [
    "ConvNeXtWrapper",
    "TIMMConvNeXtWrapper",
    "ConvNeXtWithLinearProbe",
    "ConvNeXtFeaturePyramid",
    "create_convnext_model"
]