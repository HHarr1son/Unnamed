import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class CLIPWrapper(nn.Module):
    """Frozen CLIP model wrapper for image-text alignment"""
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cuda",
        use_transformers: bool = False
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        
        if use_transformers and HF_AVAILABLE:
            self._load_hf_model(model_name)
        elif CLIP_AVAILABLE:
            self._load_openai_model(model_name)
        else:
            raise ImportError("Neither clip nor transformers available")
        
        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _load_openai_model(self, model_name: str):
        """Load OpenAI CLIP model"""
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.use_hf = False
    
    def _load_hf_model(self, model_name: str):
        """Load HuggingFace CLIP model"""
        # Map common names
        hf_names = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-L/14": "openai/clip-vit-large-patch14",
            "ViT-G/14": "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
        }
        hf_model_name = hf_names.get(model_name, model_name)
        
        self.processor = CLIPProcessor.from_pretrained(hf_model_name)
        self.model = CLIPModel.from_pretrained(hf_model_name)
        self.use_hf = True
        self.model.to(self.device)
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings
        
        Args:
            images: [B, C, H, W] or [B, T, C, H, W]
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] image features
        """
        original_shape = images.shape
        if len(original_shape) == 5:  # Video frames
            B, T = original_shape[:2]
            images = images.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        if self.use_hf:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
        else:
            # OpenAI CLIP expects preprocessed images; `self.preprocess` is a torchvision transform
            # Apply in a batched manner
            if len(images.shape) == 4:
                imgs = [self.preprocess(img) for img in images]
                batch = torch.stack(imgs).to(self.device)
            else:
                raise ValueError("images must be 4D (B, C, H, W) or 5D (B, T, C, H, W)")
            image_features = self.model.encode_image(batch)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if len(original_shape) == 5:
            image_features = image_features.view(B, T, -1)  # [B, T, D]
        
        return image_features
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            torch.Tensor: [B, D] text features
        """
        if self.use_hf:
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
        else:
            text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
            text_features = self.model.encode_text(text_tokens)
        
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    @torch.no_grad()
    def compute_similarity(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        """Compute image-text similarity scores
        
        Args:
            images: [B, C, H, W]
            texts: List of B text strings
            
        Returns:
            torch.Tensor: [B, B] similarity matrix
        """
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        
        # Compute cosine similarity
        similarity = image_features @ text_features.T
        
        return similarity
    
    @property
    def embed_dim(self) -> int:
        """Get embedding dimension"""
        if hasattr(self.model, 'text_projection'):
            return self.model.text_projection.shape[1]
        elif hasattr(self.model, 'config'):
            return self.model.config.projection_dim
        else:
            return 512  # Default


class CLIPViTG14(CLIPWrapper):
    """CLIP ViT-G/14 specialized wrapper"""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("ViT-G/14", device, use_transformers=True, **kwargs)


class CLIPViTL14(CLIPWrapper):
    """CLIP ViT-L/14 specialized wrapper"""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("ViT-L/14", device, **kwargs)


def create_clip_model(
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    use_transformers: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function for CLIP models
    
    Args:
        model_name: CLIP model name
        device: Device to load model on
        use_transformers: Use HuggingFace instead of OpenAI implementation
        
    Returns:
        CLIP model wrapper
    """
    
    # Special cases for paper models
    if model_name == "ViT-G/14":
        return CLIPViTG14(device, **kwargs)
    elif model_name == "ViT-L/14":
        return CLIPViTL14(device, **kwargs)
    else:
        return CLIPWrapper(model_name, device, use_transformers, **kwargs)


__all__ = ["CLIPWrapper", "CLIPViTG14", "CLIPViTL14", "create_clip_model"]