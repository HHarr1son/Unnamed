import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import SwinModel, AutoImageProcessor, SwinConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class SwinWrapper(nn.Module):
    """Swin Transformer wrapper for hierarchical feature extraction"""
    
    def __init__(
        self,
        model_name: str = "microsoft/swin-base-patch4-window7-224",
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
        self.model = SwinModel.from_pretrained(model_name)
        
        # Get model configuration
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = len(self.config.depths)
        
        # Stage output dimensions for Swin
        self.stage_dims = [
            self.config.embed_dim * (2 ** i) 
            for i in range(self.num_layers)
        ]
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Swin Transformer
        
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
            last_hidden = outputs.last_hidden_state  # [B*T, H*W, D]
            features = last_hidden.mean(dim=1)  # [B*T, D]
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
        
        return features
    
    def extract_hierarchical_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from all Swin stages
        
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
            # hidden_state: [B, H_i*W_i, C_i] where H_i, W_i are spatial dims at stage i
            B_curr, HW, C = hidden_state.shape
            
            # Compute spatial dimensions for this stage
            # Swin reduces spatial dimensions by factor of 2 per stage (after stage 0)
            if i == 0:
                H_curr = W_curr = int((HW) ** 0.5)  # First stage maintains resolution
            else:
                # Each subsequent stage has half the spatial resolution
                total_reduction = 2 ** i
                original_h = original_w = 224  # Assume 224x224 input
                H_curr = W_curr = max(1, original_h // (4 * total_reduction))
                H_curr = W_curr = int((HW) ** 0.5)  # Use actual computed value
            
            # Reshape to spatial format: [B, C, H, W]
            if H_curr * W_curr == HW:
                spatial_features = hidden_state.transpose(1, 2).view(B_curr, C, H_curr, W_curr)
            else:
                # Fallback: use adaptive pooling
                spatial_features = hidden_state.transpose(1, 2).view(B_curr, C, -1)
                target_size = int(HW ** 0.5)
                spatial_features = spatial_features.view(B_curr, C, target_size, -1)
                spatial_features = nn.functional.adaptive_avg_pool2d(spatial_features, (H_curr, W_curr))
            
            # Global average pooling for feature vector
            pooled_features = nn.functional.adaptive_avg_pool2d(spatial_features, (1, 1))
            pooled_features = pooled_features.flatten(1)  # [B, C]
            
            features[f'stage_{i}'] = pooled_features
            features[f'stage_{i}_spatial'] = spatial_features
        
        # Reshape for video if needed
        if len(original_shape) == 5:
            for key in features:
                if 'spatial' in key:
                    _, C, H, W = features[key].shape
                    features[key] = features[key].view(B, T, C, H, W)
                else:
                    features[key] = features[key].view(B, T, -1)
        
        return features
    
    def get_patch_features(self, x: torch.Tensor, stage_idx: int = -1) -> torch.Tensor:
        """Get patch-level features from specific stage
        
        Args:
            x: [B, C, H, W] input images
            stage_idx: Which stage to extract from (-1 for last stage)
            
        Returns:
            torch.Tensor: [B, N, D] patch features
        """
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Get features from specified stage
        if stage_idx == -1:
            stage_idx = len(hidden_states) - 1
        
        patch_features = hidden_states[stage_idx]  # [B, N, D]
        return patch_features


class TIMMSwinWrapper(nn.Module):
    """TIMM Swin Transformer wrapper with more variants"""
    
    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
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
        
        # Load model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            features_only=True  # Return hierarchical features
        )
        
        # Get feature dimensions for each stage
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


class SwinFeaturePyramid(nn.Module):
    """Swin Transformer with Feature Pyramid Network"""
    
    def __init__(
        self,
        backbone_name: str = "swin_base_patch4_window7_224",
        device: str = "cuda",
        use_timm: bool = True
    ):
        super().__init__()
        
        # Load Swin backbone
        if use_timm:
            self.backbone = TIMMSwinWrapper(
                backbone_name, pretrained=True, device=device, freeze=True
            )
            stage_dims = self.backbone.stage_dims
        else:
            self.backbone = SwinWrapper(
                f"microsoft/{backbone_name.replace('_', '-')}", 
                device=device, freeze=True
            )
            stage_dims = self.backbone.stage_dims
        
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


def create_swin_model(
    model_name: str = "swin_base_patch4_window7_224",
    pretrained: bool = True,
    device: str = "cuda",
    use_timm: bool = False,
    use_fpn: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function for Swin Transformer models
    
    Args:
        model_name: Swin model variant
        pretrained: Use pretrained weights
        device: Device to load model on
        use_timm: Use TIMM implementation
        use_fpn: Use Feature Pyramid Network
        
    Returns:
        Swin Transformer model wrapper
    """
    
    if use_fpn:
        return SwinFeaturePyramid(model_name, device, use_timm, **kwargs)
    elif use_timm:
        return TIMMSwinWrapper(model_name, pretrained, device, **kwargs)
    else:
        # Convert TIMM names to HuggingFace format
        hf_name_map = {
            "swin_tiny_patch4_window7_224": "microsoft/swin-tiny-patch4-window7-224",
            "swin_small_patch4_window7_224": "microsoft/swin-small-patch4-window7-224", 
            "swin_base_patch4_window7_224": "microsoft/swin-base-patch4-window7-224",
            "swin_base_patch4_window12_384": "microsoft/swin-base-patch4-window12-384",
            "swin_large_patch4_window7_224": "microsoft/swin-large-patch4-window7-224",
            "swin_large_patch4_window12_384": "microsoft/swin-large-patch4-window12-384"
        }
        hf_name = hf_name_map.get(model_name, model_name)
        return SwinWrapper(hf_name, device, **kwargs)


__all__ = [
    "SwinWrapper",
    "TIMMSwinWrapper",
    "SwinFeaturePyramid", 
    "create_swin_model"
]