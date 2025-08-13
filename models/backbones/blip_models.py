import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class BLIP2Wrapper(nn.Module):
    """Frozen BLIP-2 model wrapper for keyframe captioning and embedding"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cuda",
        max_txt_len: int = 32
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        self.device = device
        self.max_txt_len = max_txt_len
        
        # Load processor and model
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32
        )
        
        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.to(device)
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual embeddings from images
        
        Args:
            images: [B, C, H, W] or [B, T, C, H, W]
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] visual features
        """
        original_shape = images.shape
        if len(original_shape) == 5:  # Video frames
            B, T = original_shape[:2]
            images = images.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Get vision embeddings
        vision_outputs = self.model.vision_model(**inputs)
        pooled_output = vision_outputs.pooler_output  # [B*T, D]
        
        if len(original_shape) == 5:
            pooled_output = pooled_output.view(B, T, -1)  # [B, T, D]
        
        return pooled_output
    
    @torch.no_grad()
    def generate_caption(self, images: torch.Tensor) -> List[str]:
        """Generate captions for images
        
        Args:
            images: [B, C, H, W] 
            
        Returns:
            List[str]: Generated captions
        """
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_txt_len,
            num_beams=3,
            do_sample=False
        )
        
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [cap.strip() for cap in captions]
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            torch.Tensor: [B, D] text embeddings
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        text_outputs = self.model.language_model.get_input_embeddings()(inputs.input_ids)
        # Simple mean pooling
        attention_mask = inputs.attention_mask.unsqueeze(-1)
        embeddings = (text_outputs * attention_mask).sum(1) / attention_mask.sum(1)
        
        return embeddings


class BLIPWrapper(nn.Module):
    """Frozen BLIP-1 model wrapper (lighter alternative)"""
    
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cuda"
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available")
        
        self.device = device
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.to(device)
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual embeddings"""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        vision_outputs = self.model.vision_model(**inputs)
        return vision_outputs.pooler_output
    
    @torch.no_grad()
    def generate_caption(self, images: torch.Tensor) -> List[str]:
        """Generate captions"""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=20)
        captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [cap.strip() for cap in captions]


def create_blip_model(
    model_type: str = "blip2",
    model_name: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """Factory function for BLIP models
    
    Args:
        model_type: "blip2", "blip"
        model_name: Specific model name or None for defaults
        device: Device to load model on
        
    Returns:
        BLIP model wrapper
    """
    
    if model_type == "blip2":
        default_name = "Salesforce/blip2-opt-2.7b"
        return BLIP2Wrapper(model_name or default_name, device, **kwargs)
    elif model_type == "blip":
        default_name = "Salesforce/blip-image-captioning-base"
        return BLIPWrapper(model_name or default_name, device, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


__all__ = ["BLIP2Wrapper", "BLIPWrapper", "create_blip_model"]