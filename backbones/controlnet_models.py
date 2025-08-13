import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Optional, Tuple, Union

try:
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    from controlnet_aux import CannyDetector, OpenposeDetector, MidasDetector
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class ControlNetWrapper(nn.Module):
    """Frozen ControlNet wrapper for keyframe guidance"""
    
    def __init__(
        self,
        controlnet_type: str = "canny",
        model_name: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers not available. Install with: pip install diffusers controlnet-aux")
        
        self.device = device
        self.controlnet_type = controlnet_type
        
        # Default model names
        default_models = {
            "canny": "diffusers/controlnet-canny-sdxl-1.0",
            "depth": "diffusers/controlnet-depth-sdxl-1.0",
            "pose": "diffusers/controlnet-openpose-sdxl-1.0"
        }
        
        model_name = model_name or default_models.get(controlnet_type, default_models["canny"])
        
        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32
        )
        self.controlnet.to(device)
        
        # Load condition detector
        self._setup_detector()
        
        # Freeze model
        self.controlnet.eval()
        for param in self.controlnet.parameters():
            param.requires_grad = False
    
    def _setup_detector(self):
        """Setup condition detector based on type"""
        if self.controlnet_type == "canny":
            self.detector = CannyDetector()
        elif self.controlnet_type == "depth":
            self.detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
        elif self.controlnet_type == "pose":
            self.detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        else:
            self.detector = None
    
    @torch.no_grad()
    def detect_condition(self, images: torch.Tensor) -> torch.Tensor:
        """Detect control conditions from images
        
        Args:
            images: [B, C, H, W] RGB images (0-1)
            
        Returns:
            torch.Tensor: [B, C, H, W] condition maps
        """
        if self.detector is None:
            return images  # Return original if no detector
        
        B, C, H, W = images.shape
        conditions = []
        
        # Convert to PIL/numpy for detection
        for i in range(B):
            img = images[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            img = (img * 255).astype(np.uint8)
            
            if self.controlnet_type == "canny":
                condition = self.detector(img)
            elif self.controlnet_type == "depth":
                condition = self.detector(img)
            elif self.controlnet_type == "pose":
                condition = self.detector(img)
            else:
                condition = img
            
            # Convert back to tensor
            if isinstance(condition, np.ndarray):
                if len(condition.shape) == 2:  # Grayscale
                    condition = np.stack([condition] * 3, axis=-1)
                condition = torch.from_numpy(condition).permute(2, 0, 1).float() / 255.0
            
            conditions.append(condition)
        
        return torch.stack(conditions).to(self.device)
    
    @torch.no_grad()
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through ControlNet
        
        Args:
            sample: Noisy latents [B, C, H, W]
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings [B, L, D]
            controlnet_cond: Control conditions [B, C, H, W]
            conditioning_scale: Conditioning strength
            
        Returns:
            Tuple of (down_samples, mid_sample)
        """
        return self.controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            return_dict=False
        )


class CannyControlNet(ControlNetWrapper):
    """Canny edge ControlNet for structural guidance"""
    
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("canny", device=device, **kwargs)
        self.low_threshold = 100
        self.high_threshold = 200
    
    def detect_condition(self, images: torch.Tensor) -> torch.Tensor:
        """Detect canny edges with configurable thresholds"""
        B, C, H, W = images.shape
        conditions = []
        
        for i in range(B):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            
            # Convert to grayscale for canny
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
            
            # Convert to 3-channel
            edges_rgb = np.stack([edges] * 3, axis=-1)
            edges_tensor = torch.from_numpy(edges_rgb).permute(2, 0, 1).float() / 255.0
            
            conditions.append(edges_tensor)
        
        return torch.stack(conditions).to(self.device)


class MultiControlNet(nn.Module):
    """Multiple ControlNet ensemble for combined guidance"""
    
    def __init__(
        self,
        controlnet_types: List[str] = ["canny"],
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.controlnets = nn.ModuleList()
        
        for cnet_type in controlnet_types:
            try:
                controlnet = ControlNetWrapper(cnet_type, device=device, **kwargs)
                self.controlnets.append(controlnet)
            except:
                print(f"Failed to load {cnet_type} ControlNet, skipping")
    
    def detect_conditions(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Detect all control conditions"""
        conditions = []
        for controlnet in self.controlnets:
            condition = controlnet.detect_condition(images)
            conditions.append(condition)
        return conditions
    
    def forward(self, sample, timestep, encoder_hidden_states, controlnet_conds, conditioning_scales):
        """Forward with multiple ControlNets"""
        all_down_samples = []
        all_mid_samples = []
        
        for i, controlnet in enumerate(self.controlnets):
            cond = controlnet_conds[i] if i < len(controlnet_conds) else controlnet_conds[0]
            scale = conditioning_scales[i] if i < len(conditioning_scales) else conditioning_scales[0]
            
            down_samples, mid_sample = controlnet(
                sample, timestep, encoder_hidden_states, cond, scale
            )
            all_down_samples.append(down_samples)
            all_mid_samples.append(mid_sample)
        
        # Combine outputs (simple average)
        combined_down = []
        for i in range(len(all_down_samples[0])):
            combined = torch.stack([ds[i] for ds in all_down_samples]).mean(0)
            combined_down.append(combined)
        
        combined_mid = torch.stack(all_mid_samples).mean(0)
        
        return combined_down, combined_mid


def create_controlnet(
    controlnet_type: str = "canny",
    model_name: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """Factory function for ControlNet models
    
    Args:
        controlnet_type: Type of ControlNet ("canny", "depth", "pose", "multi", "mock")
        model_name: Specific model name or None for defaults
        device: Device to load model on
        
    Returns:
        ControlNet model wrapper
    """

    if controlnet_type == "canny":
        return CannyControlNet(device=device, **kwargs)
    elif controlnet_type == "multi":
        types = kwargs.pop("controlnet_types", ["canny"])
        return MultiControlNet(types, device=device, **kwargs)
    else:
        return ControlNetWrapper(controlnet_type, model_name, device, **kwargs)


__all__ = [
    "ControlNetWrapper", 
    "CannyControlNet", 
    "MultiControlNet", 
    "create_controlnet"
]