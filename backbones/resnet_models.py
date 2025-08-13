import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    import torchvision.models as models
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from transformers import AutoImageProcessor, ResNetModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ResNetWrapper(nn.Module):
    """ResNet backbone wrapper for feature extraction"""
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        device: str = "cuda",
        freeze: bool = True,
        num_classes: Optional[int] = None
    ):
        super().__init__()
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available. Install with: pip install torchvision")
        
        self.device = device
        self.model_name = model_name
        
        # Weight mappings for new torchvision API
        weight_map = {
            "resnet18": ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            "resnet34": ResNet34_Weights.IMAGENET1K_V1 if pretrained else None,
            "resnet50": ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
            "resnet101": ResNet101_Weights.IMAGENET1K_V2 if pretrained else None,
            "resnet152": ResNet152_Weights.IMAGENET1K_V2 if pretrained else None,
        }
        
        # Load model
        if model_name == "resnet18":
            self.model = models.resnet18(weights=weight_map["resnet18"])
        elif model_name == "resnet34":
            self.model = models.resnet34(weights=weight_map["resnet34"])
        elif model_name == "resnet50":
            self.model = models.resnet50(weights=weight_map["resnet50"])
        elif model_name == "resnet101":
            self.model = models.resnet101(weights=weight_map["resnet101"])
        elif model_name == "resnet152":
            self.model = models.resnet152(weights=weight_map["resnet152"])
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        
        # Get feature dimensions
        self.feature_dim = self.model.fc.in_features
        
        # Remove final classification layer for feature extraction
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Add custom classifier if specified
        if num_classes is not None:
            self.classifier = nn.Linear(self.feature_dim, num_classes)
        else:
            self.classifier = None
        
        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
        if self.classifier is not None:
            self.classifier.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet
        
        Args:
            x: [B, C, H, W] or [B, T, C, H, W] input images
            
        Returns:
            torch.Tensor: [B, D] or [B, T, D] features
        """
        original_shape = x.shape
        if len(original_shape) == 5:  # Video frames [B, T, C, H, W]
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])  # [B*T, C, H, W]
        
        # Extract features
        features = self.model(x)  # [B*T, D, 1, 1]
        features = features.flatten(1)  # [B*T, D]
        
        # Apply classifier if available
        if self.classifier is not None:
            features = self.classifier(features)
        
        # Reshape back if video input
        if len(original_shape) == 5:
            features = features.view(B, T, -1)  # [B, T, D]
        
        return features
    
    def extract_multiscale_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from different ResNet layers"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        features = {}
        
        # Forward through each stage
        x = self.model[0](x)  # conv1 + bn + relu + maxpool
        features['conv1'] = x  # [B, 64, H/4, W/4]
        
        x = self.model[1](x)  # layer1
        features['layer1'] = x  # [B, 64/256, H/4, W/4]
        
        x = self.model[2](x)  # layer2
        features['layer2'] = x  # [B, 128/512, H/8, W/8]
        
        x = self.model[3](x)  # layer3
        features['layer3'] = x  # [B, 256/1024, H/16, W/16]
        
        x = self.model[4](x)  # layer4
        features['layer4'] = x  # [B, 512/2048, H/32, W/32]
        
        x = self.model[5](x)  # avgpool
        features['pool'] = x.flatten(1)  # [B, 512/2048]
        
        # Reshape for video if needed
        if len(original_shape) == 5:
            for key in features:
                if features[key].dim() == 4:  # Spatial features
                    _, C, H, W = features[key].shape
                    features[key] = features[key].view(B, T, C, H, W)
                else:  # Pooled features
                    features[key] = features[key].view(B, T, -1)
        
        return features


class HFResNetWrapper(nn.Module):
    """HuggingFace ResNet wrapper"""
    
    def __init__(
        self,
        model_name: str = "microsoft/resnet-50",
        device: str = "cuda",
        freeze: bool = True
    ):
        super().__init__()
        if not HF_AVAILABLE:
            raise ImportError("transformers not available")
        
        self.device = device
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ResNetModel.from_pretrained(model_name)
        
        # Freeze if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.model.to(device)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through HF ResNet"""
        original_shape = x.shape
        if len(original_shape) == 5:
            B, T = original_shape[:2]
            x = x.view(-1, *original_shape[2:])
        
        # Process images (HF expects 0-1 range)
        if x.max() > 1.0:
            x = x / 255.0
        
        # Forward through model
        outputs = self.model(pixel_values=x)
        features = outputs.pooler_output  # [B, D]
        
        if len(original_shape) == 5:
            features = features.view(B, T, -1)
        
        return features


class ResNetFeaturePyramid(nn.Module):
    """ResNet with Feature Pyramid Network for multi-scale features"""
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        
        # Load ResNet backbone
        self.backbone = ResNetWrapper(
            backbone, pretrained=pretrained, device=device, freeze=True
        )
        
        # Get channel dimensions for FPN
        if backbone in ["resnet18", "resnet34"]:
            self.channels = [64, 128, 256, 512]
        else:
            self.channels = [256, 512, 1024, 2048]
        
        # FPN lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, 256, 1) for c in self.channels
        ])
        
        # FPN output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1) for _ in self.channels
        ])
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward with FPN features"""
        # Extract backbone features
        backbone_features = self.backbone.extract_multiscale_features(x)
        
        # Use layer features for FPN
        features = [
            backbone_features['layer1'],
            backbone_features['layer2'], 
            backbone_features['layer3'],
            backbone_features['layer4']
        ]
        
        # Build FPN top-down
        fpn_features = {}
        
        # Start from highest resolution
        prev_features = None
        for i in range(len(features) - 1, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            
            if prev_features is not None:
                # Upsample and add
                lateral += nn.functional.interpolate(
                    prev_features, scale_factor=2, mode='bilinear', align_corners=False
                )
            
            fpn_features[f'p{i+2}'] = self.fpn_convs[i](lateral)
            prev_features = fpn_features[f'p{i+2}']
        
        return fpn_features


def create_resnet_model(
    model_name: str = "resnet50",
    pretrained: bool = True,
    device: str = "cuda",
    use_hf: bool = False,
    use_fpn: bool = False,
    **kwargs
) -> nn.Module:
    """Factory function for ResNet models
    
    Args:
        model_name: ResNet variant name
        pretrained: Use pretrained weights
        device: Device to load model on
        use_hf: Use HuggingFace implementation
        use_fpn: Use Feature Pyramid Network
        
    Returns:
        ResNet model wrapper
    """
    
    if use_fpn:
        return ResNetFeaturePyramid(model_name, pretrained, device, **kwargs)
    elif use_hf:
        hf_name_map = {
            "resnet50": "microsoft/resnet-50",
            "resnet18": "microsoft/resnet-18",
            "resnet34": "microsoft/resnet-34",
            "resnet101": "microsoft/resnet-101",
            "resnet152": "microsoft/resnet-152"
        }
        hf_model = hf_name_map.get(model_name, "microsoft/resnet-50")
        return HFResNetWrapper(hf_model, device, **kwargs)
    else:
        return ResNetWrapper(model_name, pretrained, device, **kwargs)


__all__ = [
    "ResNetWrapper", 
    "HFResNetWrapper", 
    "ResNetFeaturePyramid", 
    "create_resnet_model"
]