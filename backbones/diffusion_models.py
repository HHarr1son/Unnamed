import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

try:
    from diffusers import (
        StableDiffusionXLPipeline, 
        StableDiffusionPipeline,
        UNet2DConditionModel,
        DDPMScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class DiffusionBackbone(nn.Module):
    """Diffusion backbone wrapper for task-adaptive generation"""
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers not available. Install with: pip install diffusers")
        
        self.device = device
        self.torch_dtype = torch_dtype
        
        # Load UNet from pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name, torch_dtype=torch_dtype, use_safetensors=True
        )
        
        self.unet = pipeline.unet
        self.scheduler = pipeline.scheduler
        self.vae = pipeline.vae
        
        # Move to device
        self.unet.to(device)
        self.vae.to(device)
        
        # Store original forward for task adaptation
        self._original_forward = self.unet.forward
        
        # Task encoding
        self.task_embed_dim = 64
        self.task_embeddings = nn.Embedding(2, self.task_embed_dim)  # video=0, image=1
        self.task_embeddings.to(device)
        
        # Cross-attention dimension
        self.cross_attention_dim = self.unet.config.cross_attention_dim

        # Projection from task embedding dim to cross-attention dim (created once)
        self.task_proj = None
        if self.task_embed_dim != self.cross_attention_dim:
            self.task_proj = nn.Linear(self.task_embed_dim, self.cross_attention_dim).to(self.device)
    
    def encode_task(self, task_type: str, batch_size: int) -> torch.Tensor:
        """Encode task type to embeddings
        
        Args:
            task_type: "video" or "image"
            batch_size: Batch size
            
        Returns:
            torch.Tensor: [B, D] task embeddings
        """
        task_id = 0 if task_type == "video" else 1
        task_ids = torch.full((batch_size,), task_id, device=self.device)
        return self.task_embeddings(task_ids)
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        encoder_hidden_states: torch.Tensor,
        task_type: str = "image",
        **kwargs
    ) -> torch.Tensor:
        """Task-adaptive forward pass
        
        Args:
            sample: Noisy latents [B, C, H, W]
            timestep: Diffusion timestep
            encoder_hidden_states: Cross-attention features [B, L, D]
            task_type: "video" or "image"
            
        Returns:
            torch.Tensor: Predicted noise
        """
        batch_size = sample.shape[0]
        
        # Add task conditioning to encoder states
        task_embeds = self.encode_task(task_type, batch_size)
        task_embeds = task_embeds.unsqueeze(1)  # [B, 1, D]
        
        # Concatenate with existing encoder states
        if self.task_proj is not None:
            task_embeds = self.task_proj(task_embeds)
        
        enhanced_encoder_states = torch.cat([encoder_hidden_states, task_embeds], dim=1)
        
        # Forward through UNet
        return self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=enhanced_encoder_states,
            **kwargs
        ).sample
    
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using VAE
        
        Args:
            images: [B, C, H, W] images (0-1 range)
            
        Returns:
            torch.Tensor: [B, C', H', W'] latents
        """
        # Scale to [-1, 1] for VAE
        images = 2.0 * images - 1.0
        return self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using VAE
        
        Args:
            latents: [B, C', H', W'] latents
            
        Returns:
            torch.Tensor: [B, C, H, W] images (0-1 range)
        """
        latents = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents).sample
        
        # Scale to [0, 1]
        images = (images + 1.0) / 2.0
        return images.clamp(0, 1)


class StableDiffusionXLWrapper(nn.Module):
    """SDXL pipeline wrapper for unified generation"""
    
    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        enable_cpu_offload: bool = False
    ):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers not available")
        
        self.device = device
        
        # Load pipeline
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            use_safetensors=True
        )
        
        if enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(device)
        
        # Default generation params
        self.default_params = {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        }
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate images from text prompts
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s)
            **kwargs: Additional generation parameters
            
        Returns:
            torch.Tensor: [B, C, H, W] generated images
        """
        # Merge with default params
        params = {**self.default_params, **kwargs}
        
        # Generate
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **params
        )
        
        # Convert to tensor
        images = []
        for img in result.images:
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            images.append(img_tensor)
        
        return torch.stack(images)


class T2VDiffusionWrapper(nn.Module):
    """Text-to-Video diffusion wrapper"""
    
    def __init__(
        self,
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda"
    ):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers not available")
        
        self.device = device
        
        # Use base SDXL but adapt for video generation
        self.backbone = DiffusionBackbone(base_model, device)
        
        # Video-specific temporal layers (simplified)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.backbone.cross_attention_dim,
            num_heads=8,
            batch_first=True
        ).to(device)
        
        self.max_frames = 16
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        encoder_hidden_states: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for video generation
        
        Args:
            sample: [B, F, C, H, W] or [B*F, C, H, W] noisy latents
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings
            
        Returns:
            torch.Tensor: Predicted noise
        """
        original_shape = sample.shape
        
        if len(original_shape) == 5:  # [B, F, C, H, W]
            B, F, C, H, W = original_shape
            sample = sample.view(B * F, C, H, W)
            
            # Expand encoder states for all frames
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, F, 1, 1)
            encoder_hidden_states = encoder_hidden_states.view(B * F, -1, encoder_hidden_states.shape[-1])
        
        # Process through backbone
        noise_pred = self.backbone(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            task_type="video",
            **kwargs
        )
        
        if len(original_shape) == 5:
            noise_pred = noise_pred.view(B, F, C, H, W)
        
        return noise_pred


class MockDiffusion(nn.Module):
    """Mock diffusion model for testing"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.cross_attention_dim = 2048
    
    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        return torch.randn_like(sample)
    
    def encode_image(self, images):
        B, C, H, W = images.shape
        return torch.randn(B, 4, H//8, W//8, device=images.device)
    
    def decode_latent(self, latents):
        B, C, H, W = latents.shape
        return torch.randn(B, 3, H*8, W*8, device=latents.device)
    
    def generate(self, prompt, **kwargs):
        if isinstance(prompt, str):
            prompt = [prompt]
        B = len(prompt)
        return torch.randn(B, 3, 512, 512)


def create_diffusion_model(
    model_type: str = "sdxl",
    model_name: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """Factory function for diffusion models
    
    Args:
        model_type: "sdxl", "sd", "t2v", "mock"
        model_name: Specific model name or None for defaults
        device: Device to load model on
        
    Returns:
        Diffusion model wrapper
    """
    if not DIFFUSERS_AVAILABLE and model_type != "mock":
        print("diffusers not available, using mock diffusion")
        return MockDiffusion(**kwargs)
    
    default_models = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd": "runwayml/stable-diffusion-v1-5",
        "t2v": "stabilityai/stable-diffusion-xl-base-1.0"
    }
    
    model_name = model_name or default_models.get(model_type, default_models["sdxl"])
    
    if model_type == "sdxl":
        return StableDiffusionXLWrapper(model_name, device, **kwargs)
    elif model_type == "t2v":
        return T2VDiffusionWrapper(model_name, device, **kwargs)
    elif model_type == "backbone":
        return DiffusionBackbone(model_name, device, **kwargs)
    elif model_type == "mock":
        return MockDiffusion(**kwargs)
    else:
        return StableDiffusionXLWrapper(model_name, device, **kwargs)


__all__ = [
    "DiffusionBackbone",
    "StableDiffusionXLWrapper", 
    "T2VDiffusionWrapper",
    "MockDiffusion",
    "create_diffusion_model"
]