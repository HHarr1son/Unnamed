import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import ViTModel, ViTImageProcessor, ViTConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ViTWrapper(nn.Module):
    """Vision Transformer wrapper for image feature extraction"""
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device: str = "cuda",
        freeze: bool = True,
        use_cls_token: bool = True,
        patch_size: int = 16,
        image_size: int = 224
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        self.device = device
        self.model_name = model_name
        self.use_cls_token = use_cls_token
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Load processor and model
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        # Get model dimensions
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through ViT
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] input images
            return_attention: Whether to return attention weights
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] features
            Optional attention weights if requested
        """
        original_shape = x.shape
        if len(original_shape) == 5:  # Video frames [B, T, C, H, W]
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        # Forward through model
        outputs = self.model(pixel_values=x, output_attentions=return_attention)
        
        if self.use_cls_token:
            # Use CLS token representation
            features = outputs.last_hidden_state[:, 0]  # [B*T, D]
        else:
            # Global average pooling over patch tokens
            patch_embeddings = outputs.last_hidden_state[:, 1:]  # [B*T, N, D]
            features = patch_embeddings.mean(dim=1)  # [B*T, D]
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
        
        if return_attention:
            attentions = outputs.attentions
            if len(original_shape) == 5:
                # Reshape attentions for video
                attentions = [att.view(B, T, *att.shape[1:]) for att in attentions]
            return features, attentions
        
        return features
    
    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features without CLS token
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            torch.Tensor: [B, N, D] patch features where N = (H/P) * (W/P)
        """
        outputs = self.model(pixel_values=x)
        patch_features = outputs.last_hidden_state[:, 1:]  # Remove CLS token
        return patch_features
    
    def extract_multiscale_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from multiple transformer layers"""
        outputs = self.model(pixel_values=x, output_hidden_states=True)
        
        features = {}
        hidden_states = outputs.hidden_states
        
        # Extract features from different layers
        layer_indices = [3, 6, 9, 12] if len(hidden_states) >= 12 else [len(hidden_states)//4, len(hidden_states)//2, 3*len(hidden_states)//4, -1]
        
        for i, layer_idx in enumerate(layer_indices):
            if layer_idx >= len(hidden_states):
                continue
            layer_features = hidden_states[layer_idx]
            
            if self.use_cls_token:
                features[f'layer_{layer_idx}'] = layer_features[:, 0]  # [B, D]
            else:
                features[f'layer_{layer_idx}'] = layer_features[:, 1:].mean(dim=1)  # [B, D]
        
        return features


class TIMMViTWrapper(nn.Module):
    """TIMM Vision Transformer wrapper with more model variants"""
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        device: str = "cuda",
        freeze: bool = True,
        num_classes: int = 0  # 0 for feature extraction
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
        """Forward pass through TIMM ViT"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        features = self.model(x)
        
        if len(original_shape) == 5:
            features = features.view(B, T, -1)
        
        return features


class ViTWithLinearProbe(nn.Module):
    """ViT with linear probing head for downstream tasks"""
    
    def __init__(
        self,
        backbone_name: str = "google/vit-base-patch16-224",
        num_classes: int = 1000,
        device: str = "cuda",
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # Load frozen backbone
        self.backbone = ViTWrapper(
            backbone_name, device=device, freeze=freeze_backbone
        )
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.backbone.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.backbone.hidden_size, num_classes)
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with classification head"""
        features = self.backbone(x)
        return self.classifier(features)


class ViTMAE(nn.Module):
    """Vision Transformer for Masked Autoencoding (simplified version)"""
    
    def __init__(
        self,
        base_model: str = "google/vit-base-patch16-224",
        device: str = "cuda",
        mask_ratio: float = 0.75
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.device = device
        
        # Load base ViT
        self.vit = ViTWrapper(base_model, device=device, freeze=False)
        
        # Decoder for reconstruction (simplified)
        self.decoder = nn.Sequential(
            nn.Linear(self.vit.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.vit.patch_size ** 2 * 3)  # Reconstruct RGB patches
        ).to(device)
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking to patch embeddings"""
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get random indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep only subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Create binary mask: 0 for kept, 1 for removed
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with masking and reconstruction"""
        # Get patch embeddings (without CLS token)
        patch_features = self.vit.extract_patch_features(x)  # [B, N, D]
        
        # Apply masking
        masked_features, mask, ids_restore = self.random_masking(patch_features)
        
        # Encode masked patches
        # (In practice, you'd need to modify the ViT forward pass)
        encoded = masked_features
        
        # Decode for reconstruction
        reconstructed_patches = self.decoder(encoded)
        
        return reconstructed_patches, mask, ids_restore


def create_vit_model(
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
    device: str = "cuda",
    use_timm: bool = False,
    model_type: str = "standard",
    **kwargs
) -> nn.Module:
    """Factory function for Vision Transformer models
    
    Args:
        model_name: Model name/variant
        pretrained: Use pretrained weights
        device: Device to load model on
        use_timm: Use TIMM implementation
        model_type: "standard", "probe", "mae"
        
    Returns:
        Vision Transformer model wrapper
    """
    
    if model_type == "mae":
        hf_name_map = {
            "vit_base_patch16_224": "google/vit-base-patch16-224",
            "vit_large_patch16_224": "google/vit-large-patch16-224",
            "vit_huge_patch14_224": "google/vit-huge-patch14-224-in21k"
        }
        hf_name = hf_name_map.get(model_name, "google/vit-base-patch16-224")
        return ViTMAE(hf_name, device, **kwargs)
    
    elif model_type == "probe":
        hf_name_map = {
            "vit_base_patch16_224": "google/vit-base-patch16-224",
            "vit_large_patch16_224": "google/vit-large-patch16-224"
        }
        hf_name = hf_name_map.get(model_name, "google/vit-base-patch16-224")
        return ViTWithLinearProbe(hf_name, device=device, **kwargs)
    
    elif use_timm:
        return TIMMViTWrapper(model_name, pretrained, device, **kwargs)
    
    else:
        # HuggingFace implementation
        hf_name_map = {
            "vit_base_patch16_224": "google/vit-base-patch16-224",
            "vit_large_patch16_224": "google/vit-large-patch16-224",
            "vit_huge_patch14_224": "google/vit-huge-patch14-224-in21k",
            "vit_base_patch32_224": "google/vit-base-patch32-224"
        }
        hf_name = hf_name_map.get(model_name, model_name)
        return ViTWrapper(hf_name, device, **kwargs)


__all__ = [
    "ViTWrapper",
    "TIMMViTWrapper", 
    "ViTWithLinearProbe",
    "ViTMAE",
    "create_vit_model"
]