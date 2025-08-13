import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import AutoImageProcessor, Dinov2Model, Dinov2Config
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class DINOv2Wrapper(nn.Module):
    """DINOv2 wrapper for self-supervised feature extraction"""
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
        freeze: bool = True,
        use_cls_token: bool = True,
        return_patch_tokens: bool = False
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        self.device = device
        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self.return_patch_tokens = return_patch_tokens
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)
        
        # Get model configuration
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.patch_size = self.config.patch_size
        self.image_size = getattr(self.config, 'image_size', 224)
        
        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through DINOv2
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] input images
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] features
            or Tuple of (cls_tokens, patch_tokens) if return_patch_tokens=True
        """
        original_shape = x.shape
        if len(original_shape) == 5:  # Video frames [B, T, C, H, W]
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        # Forward through model
        with torch.no_grad() if self.model.training == False else torch.enable_grad():
            outputs = self.model(pixel_values=x)
            
            # Extract CLS token and patch tokens
            last_hidden_state = outputs.last_hidden_state  # [B*T, 1+N, D]
            cls_tokens = last_hidden_state[:, 0]  # [B*T, D]
            patch_tokens = last_hidden_state[:, 1:]  # [B*T, N, D]
        
        # Choose output based on configuration
        if self.use_cls_token:
            features = cls_tokens
        else:
            # Global average pooling over patches
            features = patch_tokens.mean(dim=1)  # [B*T, D]
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
            if self.return_patch_tokens:
                patch_tokens = patch_tokens.view(B, T, self.num_patches, -1)
        
        if self.return_patch_tokens:
            return features, patch_tokens
        
        return features
    
    def extract_patch_features(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Extract patch-level features
        
        Args:
            x: [B, C, H, W] input images
            layer_idx: Which layer to extract from (-1 for last layer)
            
        Returns:
            torch.Tensor: [B, N, D] patch features
        """
        if layer_idx == -1:
            # Use last layer
            outputs = self.model(pixel_values=x)
            patch_features = outputs.last_hidden_state[:, 1:]  # Skip CLS token
        else:
            # Extract from specific layer
            outputs = self.model(pixel_values=x, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            patch_features = hidden_states[layer_idx][:, 1:]
        
        return patch_features
    
    def extract_multiscale_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from multiple transformer layers
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            Dict with features from different layers
        """
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        # Forward with all hidden states
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        features = {}
        
        # Extract features from key layers
        layer_indices = [3, 6, 9, 12] if len(hidden_states) >= 12 else [len(hidden_states)//4, len(hidden_states)//2, 3*len(hidden_states)//4, -1]
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(hidden_states):
                continue
                
            layer_features = hidden_states[layer_idx]
            
            # Extract CLS and patch tokens separately
            cls_tokens = layer_features[:, 0]  # [B*T, D]
            patch_tokens = layer_features[:, 1:]  # [B*T, N, D]
            
            # Global features
            if self.use_cls_token:
                global_features = cls_tokens
            else:
                global_features = patch_tokens.mean(dim=1)
            
            features[f'layer_{layer_idx}_global'] = global_features
            features[f'layer_{layer_idx}_patches'] = patch_tokens
        
        # Reshape for video if needed
        if len(original_shape) == 5:
            for key in features:
                if 'patches' in key:
                    _, N, D = features[key].shape
                    features[key] = features[key].view(B, T, N, D)
                else:
                    features[key] = features[key].view(B, T, -1)
        
        return features
    
    def compute_patch_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """Compute patch-to-patch similarity matrix using DINOv2 features
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            torch.Tensor: [B, N, N] similarity matrices
        """
        patch_features = self.extract_patch_features(x)  # [B, N, D]
        
        # Normalize features
        patch_features = nn.functional.normalize(patch_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.bmm(patch_features, patch_features.transpose(1, 2))  # [B, N, N]
        
        return similarity


class TIMMDINOv2Wrapper(nn.Module):
    """TIMM DINOv2 wrapper with additional variants"""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
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
            num_classes=num_classes
        )
        
        # Get feature dimension
        if hasattr(self.model, 'embed_dim'):
            self.feature_dim = self.model.embed_dim
        elif hasattr(self.model, 'num_features'):
            self.feature_dim = self.model.num_features
        else:
            self.feature_dim = 768  # Default
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TIMM DINOv2"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        features = self.model(x)
        
        if len(original_shape) == 5:
            features = features.view(B, T, -1)
        
        return features


class DINOv2WithLinearProbe(nn.Module):
    """DINOv2 with linear probing head for downstream tasks"""
    
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        num_classes: int = 1000,
        device: str = "cuda",
        freeze_backbone: bool = True,
        dropout_rate: float = 0.0,
        use_norm: bool = True
    ):
        super().__init__()
        
        # Load frozen backbone
        self.backbone = DINOv2Wrapper(
            backbone_name, device=device, freeze=freeze_backbone
        )
        
        # Build classifier
        layers = []
        
        if use_norm:
            layers.append(nn.LayerNorm(self.backbone.hidden_size))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(self.backbone.hidden_size, num_classes))
        
        self.classifier = nn.Sequential(*layers).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with classification head"""
        features = self.backbone(x)
        return self.classifier(features)


class DINOv2ForSegmentation(nn.Module):
    """DINOv2 adapted for dense prediction tasks like segmentation"""
    
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        num_classes: int = 21,
        device: str = "cuda",
        upsampling_factor: int = 4
    ):
        super().__init__()
        
        # Load DINOv2 backbone
        self.backbone = DINOv2Wrapper(
            backbone_name, device=device, freeze=False, return_patch_tokens=True
        )
        
        self.upsampling_factor = upsampling_factor
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.backbone.hidden_size, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        ).to(device)
        
        # Upsampling layer
        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes, 
            kernel_size=upsampling_factor, 
            stride=upsampling_factor
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation"""
        B, C, H, W = x.shape
        
        # Extract patch features
        _, patch_tokens = self.backbone(x)  # [B, N, D]
        
        # Reshape patch tokens to spatial format
        patch_size = self.backbone.patch_size
        H_patches = W_patches = int((patch_tokens.shape[1]) ** 0.5)
        
        # [B, N, D] -> [B, D, H_patch, W_patch]
        spatial_features = patch_tokens.transpose(1, 2).reshape(
            B, self.backbone.hidden_size, H_patches, W_patches
        )
        
        # Apply segmentation head
        seg_logits = self.seg_head(spatial_features)
        
        # Upsample to original resolution
        output = self.upsample(seg_logits)
        
        # Final resize to exact input size if needed
        if output.shape[-2:] != (H, W):
            output = nn.functional.interpolate(
                output, size=(H, W), mode='bilinear', align_corners=False
            )
        
        return output


def create_dinov2_model(
    model_name: str = "facebook/dinov2-base",
    model_type: str = "standard",
    device: str = "cuda",
    use_timm: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function for DINOv2 models
    
    Args:
        model_name: DINOv2 model variant
        model_type: "standard", "probe", "segmentation"
        device: Device to load model on
        use_timm: Use TIMM implementation
        
    Returns:
        DINOv2 model wrapper
    """
    
    if model_type == "probe":
        return DINOv2WithLinearProbe(model_name, device=device, **kwargs)
    elif model_type == "segmentation":
        return DINOv2ForSegmentation(model_name, device=device, **kwargs)
    elif use_timm:
        # Convert HF names to TIMM format
        timm_name_map = {
            "facebook/dinov2-small": "vit_small_patch14_dinov2.lvd142m",
            "facebook/dinov2-base": "vit_base_patch14_dinov2.lvd142m",
            "facebook/dinov2-large": "vit_large_patch14_dinov2.lvd142m",
            "facebook/dinov2-giant": "vit_giant_patch14_dinov2.lvd142m"
        }
        timm_name = timm_name_map.get(model_name, "vit_base_patch14_dinov2.lvd142m")
        return TIMMDINOv2Wrapper(timm_name, device=device, **kwargs)
    else:
        return DINOv2Wrapper(model_name, device, **kwargs)


# Available DINOv2 model variants
DINOV2_MODELS = {
    "dinov2_small": "facebook/dinov2-small",
    "dinov2_base": "facebook/dinov2-base", 
    "dinov2_large": "facebook/dinov2-large",
    "dinov2_giant": "facebook/dinov2-giant"
}


__all__ = [
    "DINOv2Wrapper",
    "TIMMDINOv2Wrapper",
    "DINOv2WithLinearProbe",
    "DINOv2ForSegmentation", 
    "create_dinov2_model",
    "DINOV2_MODELS"
]