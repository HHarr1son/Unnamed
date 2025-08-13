import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import SamModel, SamProcessor
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class SAMWrapper(nn.Module):
    """Segment Anything Model (SAM) wrapper for segmentation and feature extraction"""
    
    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cuda",
        freeze: bool = True
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        self.device = device
        self.model_name = model_name
        
        # Load processor and model
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name)
        
        # Get model configuration
        self.config = self.model.config
        
        # Get vision encoder (ViT backbone)
        self.vision_encoder = self.model.vision_encoder
        self.hidden_size = self.vision_encoder.config.hidden_size
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Extract image features using SAM's vision encoder
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] input images
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] image features
        """
        original_shape = x.shape
        if len(original_shape) == 5:  # Video frames [B, T, C, H, W]
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        # Process images through vision encoder
        with torch.no_grad():
            # SAM expects pixel values in specific format
            if x.max() <= 1.0:
                x = x * 255.0  # Convert to 0-255 range
            
            # Get image embeddings
            image_embeddings = self.model.get_image_embeddings(pixel_values=x)
            
            # Global average pooling to get feature vectors
            features = image_embeddings.mean(dim=(-2, -1))  # [B*T, D]
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning image features"""
        return self.encode_image(x)
    
    def segment_image(
        self,
        image: torch.Tensor,
        input_points: Optional[np.ndarray] = None,
        input_labels: Optional[np.ndarray] = None,
        input_boxes: Optional[np.ndarray] = None,
        multimask_output: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Generate segmentation masks for image
        
        Args:
            image: [C, H, W] input image
            input_points: [N, 2] array of point prompts
            input_labels: [N] array of point labels (1=foreground, 0=background)
            input_boxes: [N, 4] array of bounding box prompts
            multimask_output: Whether to return multiple masks
            
        Returns:
            Dict with masks, scores, and logits
        """
        # Convert tensor to numpy for processor
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).cpu().numpy()
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        # Prepare inputs
        inputs = {"images": image_np}
        
        if input_points is not None:
            inputs["input_points"] = [input_points]
        if input_labels is not None:
            inputs["input_labels"] = [input_labels]
        if input_boxes is not None:
            inputs["input_boxes"] = [input_boxes]
        
        # Process inputs
        processed = self.processor(**inputs, return_tensors="pt")
        
        # Move to device
        processed = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in processed.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**processed, multimask_output=multimask_output)
        
        return {
            'masks': outputs.pred_masks,
            'scores': outputs.iou_scores,
            'logits': outputs.pred_masks  # Same as masks for compatibility
        }
    
    def extract_multiscale_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from SAM vision encoder
        
        Args:
            x: [B, C, H, W] input images
            
        Returns:
            Dict with features from different layers
        """
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        features = {}
        
        # Get vision encoder features at different scales
        with torch.no_grad():
            if x.max() <= 1.0:
                x = x * 255.0
            
            # Forward through vision encoder with output_hidden_states
            vision_outputs = self.vision_encoder(
                pixel_values=x, 
                output_hidden_states=True
            )
            
            hidden_states = vision_outputs.hidden_states
            
            # Extract features from different layers
            layer_indices = [3, 6, 9, 12] if len(hidden_states) >= 12 else range(0, len(hidden_states), len(hidden_states)//4)
            
            for i, layer_idx in enumerate(layer_indices):
                if layer_idx < len(hidden_states):
                    layer_features = hidden_states[layer_idx]
                    
                    # Global average pooling
                    if len(layer_features.shape) == 3:  # [B, N, D]
                        pooled_features = layer_features.mean(dim=1)
                    else:  # [B, H, W, D]
                        pooled_features = layer_features.mean(dim=(1, 2))
                    
                    features[f'layer_{layer_idx}'] = pooled_features
        
        # Reshape for video if needed
        if len(original_shape) == 5:
            for key in features:
                features[key] = features[key].view(B, T, -1)
        
        return features


class SAMFeatureExtractor(nn.Module):
    """SAM-based feature extractor for downstream tasks"""
    
    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cuda",
        output_dim: int = 256,
        use_neck: bool = True
    ):
        super().__init__()
        
        # Load SAM backbone
        self.sam = SAMWrapper(model_name, device=device, freeze=True)
        
        self.use_neck = use_neck
        
        if use_neck:
            # Add projection neck
            self.neck = nn.Sequential(
                nn.Linear(self.sam.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, output_dim)
            ).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with optional neck"""
        features = self.sam.encode_image(x)
        
        if self.use_neck:
            features = self.neck(features)
        
        return features


class SAMForMaskedImageModeling(nn.Module):
    """SAM adapted for masked image modeling tasks"""
    
    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cuda",
        mask_ratio: float = 0.75
    ):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        
        # Load SAM
        self.sam = SAMWrapper(model_name, device=device, freeze=False)
        
        # Add reconstruction head
        self.decoder = nn.Sequential(
            nn.Linear(self.sam.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 16 * 16 * 3)  # Assume 16x16 patches
        ).to(device)
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to image patches"""
        B, C, H, W = x.shape
        
        # Simple grid-based masking
        patch_size = 16
        H_patches, W_patches = H // patch_size, W // patch_size
        total_patches = H_patches * W_patches
        
        num_masked = int(total_patches * self.mask_ratio)
        
        # Create random mask
        mask = torch.zeros(B, total_patches, device=x.device)
        for b in range(B):
            masked_indices = torch.randperm(total_patches)[:num_masked]
            mask[b, masked_indices] = 1
        
        return mask
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with reconstruction loss"""
        # Generate mask
        mask = self.random_masking(x)
        
        # Extract features
        features = self.sam.encode_image(x)
        
        # Reconstruct (simplified)
        reconstructed = self.decoder(features)
        
        return {
            'features': features,
            'reconstructed': reconstructed,
            'mask': mask
        }


def create_sam_model(
    model_name: str = "facebook/sam-vit-base",
    model_type: str = "standard",
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """Factory function for SAM models
    
    Args:
        model_name: SAM model variant
        model_type: "standard", "feature_extractor", "masked_modeling"
        device: Device to load model on
        
    Returns:
        SAM model wrapper
    """
    
    if model_type == "feature_extractor":
        return SAMFeatureExtractor(model_name, device, **kwargs)
    elif model_type == "masked_modeling":
        return SAMForMaskedImageModeling(model_name, device, **kwargs)
    else:
        return SAMWrapper(model_name, device, **kwargs)


# Available SAM model variants
SAM_MODELS = {
    "sam_vit_base": "facebook/sam-vit-base",
    "sam_vit_large": "facebook/sam-vit-large",
    "sam_vit_huge": "facebook/sam-vit-huge"
}


# Utility functions for SAM prompts
def points_to_sam_format(points: List[Tuple[int, int]], labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert point lists to SAM format
    
    Args:
        points: List of (x, y) coordinates
        labels: List of labels (1=foreground, 0=background)
        
    Returns:
        Tuple of (point_array, label_array)
    """
    point_array = np.array(points, dtype=np.float32)
    label_array = np.array(labels, dtype=np.int32)
    
    return point_array, label_array


def box_to_sam_format(boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Convert bounding boxes to SAM format
    
    Args:
        boxes: List of (x1, y1, x2, y2) bounding boxes
        
    Returns:
        Box array in SAM format
    """
    return np.array(boxes, dtype=np.float32)


__all__ = [
    "SAMWrapper",
    "SAMFeatureExtractor",
    "SAMForMaskedImageModeling",
    "create_sam_model",
    "SAM_MODELS",
    "points_to_sam_format",
    "box_to_sam_format"
]