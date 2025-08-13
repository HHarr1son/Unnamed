import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import ViTMAEModel, AutoImageProcessor, ViTMAEConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class MAEWrapper(nn.Module):
    """Masked Autoencoder (MAE) wrapper for self-supervised feature extraction"""
    
    def __init__(
        self,
        model_name: str = "facebook/vit-mae-base",
        device: str = "cuda",
        freeze: bool = True,
        mask_ratio: float = 0.75,
        use_cls_token: bool = True
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        self.device = device
        self.model_name = model_name
        self.mask_ratio = mask_ratio
        self.use_cls_token = use_cls_token
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTMAEModel.from_pretrained(model_name)
        
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
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict]:
        """Forward pass through MAE
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] input images
            mask: Optional mask tensor [B, N] where N is num_patches
            return_dict: Whether to return detailed outputs
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] features
            or Dict with detailed outputs if return_dict=True
        """
        original_shape = x.shape
        if len(original_shape) == 5:  # Video frames [B, T, C, H, W]
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])  # [B*T, C, H, W]
            if mask is not None:
                mask = mask.view(-1, mask.shape[-1])  # [B*T, N]
        
        # Forward through model
        if mask is not None:
            outputs = self.model(pixel_values=x, bool_masked_pos=mask)
        else:
            outputs = self.model(pixel_values=x)
        
        # Extract features
        if self.use_cls_token:
            features = outputs.last_hidden_state[:, 0]  # CLS token [B*T, D]
        else:
            # Average over visible patches only
            if hasattr(outputs, 'ids_restore') and outputs.ids_restore is not None:
                visible_patches = outputs.last_hidden_state  # Only visible patches
                features = visible_patches.mean(dim=1)  # [B*T, D]
            else:
                features = outputs.last_hidden_state[:, 1:].mean(dim=1)  # Skip CLS, average patches
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
        
        if return_dict:
            result = {
                'features': features,
                'last_hidden_state': outputs.last_hidden_state,
                'logits': getattr(outputs, 'logits', None),
                'mask': getattr(outputs, 'mask', mask),
                'ids_restore': getattr(outputs, 'ids_restore', None)
            }
            return result
        
        return features
    
    def extract_patch_features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract patch-level features
        
        Args:
            x: [B, C, H, W] input images
            mask: Optional mask [B, N]
            
        Returns:
            torch.Tensor: [B, N_visible, D] visible patch features
        """
        if mask is not None:
            outputs = self.model(pixel_values=x, bool_masked_pos=mask)
        else:
            outputs = self.model(pixel_values=x)
        
        return outputs.last_hidden_state
    
    def random_masking(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random mask for input
        
        Args:
            x: [B, C, H, W] input images
            mask_ratio: Fraction of patches to mask (default: self.mask_ratio)
            
        Returns:
            Tuple of (features, mask) where mask is [B, N] boolean tensor
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        B = x.shape[0]
        N = self.num_patches
        len_keep = int(N * (1 - mask_ratio))
        
        # Generate random mask for each sample
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Create boolean mask: True for masked patches
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask[:, :len_keep] = False
        mask = torch.gather(mask, dim=1, index=ids_shuffle)
        
        # Forward with mask
        outputs = self.forward(x, mask=mask, return_dict=True)
        
        return outputs['features'], mask
    
    def reconstruct_image(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct masked image patches
        
        Args:
            x: [B, C, H, W] input images
            mask: [B, N] boolean mask (True for masked patches)
            
        Returns:
            torch.Tensor: [B, C, H, W] reconstructed images
        """
        if mask is None:
            # Generate random mask
            _, mask = self.random_masking(x)
        
        # Forward pass
        outputs = self.model(pixel_values=x, bool_masked_pos=mask)
        
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            # Reconstruct from logits
            logits = outputs.logits  # [B, N_masked, P*P*3]
            
            # Reshape logits to patch format
            P = self.patch_size
            N_masked = logits.shape[1]
            patch_dim = P * P * 3
            
            # Reconstruct patches
            patches = logits.view(-1, N_masked, P, P, 3)
            
            # Convert back to full image (simplified version)
            B, C, H, W = x.shape
            reconstructed = torch.zeros_like(x)
            
            # This is a simplified reconstruction - in practice, you'd need
            # proper patch-to-image conversion with mask handling
            
            return reconstructed
        else:
            # Model doesn't support reconstruction, return original
            return x


class MAEForPretraining(nn.Module):
    """MAE model configured for self-supervised pretraining"""
    
    def __init__(
        self,
        encoder_model: str = "facebook/vit-mae-base",
        device: str = "cuda",
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        
        # Load MAE model
        self.mae = MAEWrapper(
            encoder_model, device=device, freeze=False, mask_ratio=mask_ratio
        )
        
        # Add reconstruction head if model doesn't have one
        if not hasattr(self.mae.model, 'decoder'):
            self.decoder = nn.Sequential(
                nn.Linear(self.mae.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.mae.patch_size ** 2 * 3)
            ).to(device)
        else:
            self.decoder = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for pretraining
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            Dict with loss, predictions, mask, etc.
        """
        # Random masking
        features, mask = self.mae.random_masking(x)
        
        # Get reconstruction if available
        if hasattr(self.mae.model, 'logits'):
            outputs = self.mae.model(pixel_values=x, bool_masked_pos=mask)
            pred = outputs.logits
        elif self.decoder is not None:
            # Use custom decoder
            patch_features = self.mae.extract_patch_features(x, mask)
            pred = self.decoder(patch_features)
        else:
            pred = None
        
        # Compute loss if prediction available
        loss = None
        if pred is not None:
            loss = self.compute_reconstruction_loss(x, pred, mask)
        
        return {
            'loss': loss,
            'pred': pred,
            'mask': mask,
            'features': features
        }
    
    def compute_reconstruction_loss(
        self, 
        imgs: torch.Tensor, 
        pred: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss for masked patches"""
        
        # Convert images to patches
        B, C, H, W = imgs.shape
        P = self.mae.patch_size
        
        # Patchify images
        imgs = imgs.reshape(B, C, H//P, P, W//P, P)
        imgs = imgs.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, P*P*C)
        
        # Normalize patches if requested
        if self.norm_pix_loss:
            mean = imgs.mean(dim=-1, keepdim=True)
            var = imgs.var(dim=-1, keepdim=True)
            imgs = (imgs - mean) / (var + 1.e-6)**.5
        
        # Compute loss only on masked patches
        loss = (pred - imgs) ** 2
        loss = loss.mean(dim=-1)  # [B, N], mean loss per patch
        
        # Apply mask - only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


class MAEFinetune(nn.Module):
    """MAE model for fine-tuning on downstream tasks"""
    
    def __init__(
        self,
        encoder_model: str = "facebook/vit-mae-base",
        num_classes: int = 1000,
        device: str = "cuda",
        freeze_encoder: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pretrained MAE encoder
        self.encoder = MAEWrapper(
            encoder_model, device=device, freeze=freeze_encoder
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.hidden_size, num_classes)
        ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification"""
        features = self.encoder(x)
        return self.classifier(features)


def create_mae_model(
    model_name: str = "facebook/vit-mae-base",
    model_type: str = "encoder",
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """Factory function for MAE models
    
    Args:
        model_name: MAE model name
        model_type: "encoder", "pretrain", "finetune"
        device: Device to load model on
        
    Returns:
        MAE model wrapper
    """
    
    if model_type == "pretrain":
        return MAEForPretraining(model_name, device, **kwargs)
    elif model_type == "finetune":
        return MAEFinetune(model_name, device=device, **kwargs)
    else:
        return MAEWrapper(model_name, device, **kwargs)


# Available MAE model variants
MAE_MODELS = {
    "mae_vit_base": "facebook/vit-mae-base",
    "mae_vit_large": "facebook/vit-mae-large", 
    "mae_vit_huge": "facebook/vit-mae-huge"
}


__all__ = [
    "MAEWrapper",
    "MAEForPretraining", 
    "MAEFinetune",
    "create_mae_model",
    "MAE_MODELS"
]