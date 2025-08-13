"""
LLaMA Models Integration for NEED Framework

Simple wrappers for LLaMA models with LoRA adaptation for EEG-to-text generation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

try:
    from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LLaMAAdapter(nn.Module):
    """LLaMA model with LoRA adapter for EEG conditioning"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        max_length: int = 256
    ):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers and peft not available. Install with: pip install transformers peft")
        
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map="auto" if "cuda" in device else None
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # EEG conditioning layer
        self.eeg_embed_dim = 256
        self.hidden_dim = self.model.config.hidden_size
        self.eeg_projection = nn.Linear(self.eeg_embed_dim, self.hidden_dim).to(device)
        
        # Special tokens for EEG conditioning
        self.eeg_start_token = "<EEG>"
        self.eeg_end_token = "</EEG>"
        
        # Add special tokens to tokenizer
        special_tokens = {"additional_special_tokens": [self.eeg_start_token, self.eeg_end_token]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def encode_eeg_prompt(self, eeg_features: torch.Tensor, text_prompt: str = "Describe the visual content:") -> Dict:
        """Encode EEG features with text prompt
        
        Args:
            eeg_features: [B, D] EEG embeddings
            text_prompt: Text prompt for conditioning
            
        Returns:
            Dict with input_ids, attention_mask, and eeg_embeds
        """
        batch_size = eeg_features.shape[0]
        
        # Project EEG features
        eeg_embeds = self.eeg_projection(eeg_features)  # [B, hidden_dim]
        
        # Create prompt with EEG placeholder
        full_prompt = f"{text_prompt} {self.eeg_start_token} {self.eeg_end_token}"
        
        # Tokenize
        inputs = self.tokenizer(
            [full_prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "eeg_embeds": eeg_embeds
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        eeg_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with EEG conditioning
        
        Args:
            input_ids: [B, L] token ids
            attention_mask: [B, L] attention mask
            eeg_embeds: [B, D] EEG embeddings
            labels: [B, L] target labels for training
            
        Returns:
            Model outputs (logits or loss)
        """
        # Get input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Inject EEG embeddings at EEG token positions
        if eeg_embeds is not None:
            eeg_token_id = self.tokenizer.convert_tokens_to_ids(self.eeg_start_token)
            eeg_positions = (input_ids == eeg_token_id).nonzero(as_tuple=True)
            
            if len(eeg_positions[0]) > 0:
                for i, (batch_idx, seq_idx) in enumerate(zip(*eeg_positions)):
                    if i < eeg_embeds.shape[0]:
                        inputs_embeds[batch_idx, seq_idx] = eeg_embeds[i]
        
        # Forward through model
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
    
    @torch.no_grad()
    def generate_text(
        self,
        eeg_features: torch.Tensor,
        prompt: str = "Describe the visual content:",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> List[str]:
        """Generate text descriptions from EEG features
        
        Args:
            eeg_features: [B, D] EEG embeddings
            prompt: Text prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            List of generated text descriptions
        """
        encoded = self.encode_eeg_prompt(eeg_features, prompt)

        # Build inputs_embeds and inject EEG embeddings at the EEG token position
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        eeg_embeds = encoded["eeg_embeds"]

        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        eeg_token_id = self.tokenizer.convert_tokens_to_ids(self.eeg_start_token)
        eeg_positions = (input_ids == eeg_token_id).nonzero(as_tuple=True)
        if len(eeg_positions[0]) > 0:
            for i, (batch_idx, seq_idx) in enumerate(zip(*eeg_positions)):
                if i < eeg_embeds.shape[0]:
                    inputs_embeds[batch_idx, seq_idx] = eeg_embeds[i]

        # Generate
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode generated text
        generated_texts = []
        for output in outputs:
            # Skip input tokens
            new_tokens = output[encoded["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            generated_texts.append(text.strip())
        
        return generated_texts


class LightLLaMA(nn.Module):
    """Lightweight LLaMA wrapper for fast inference"""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available")
        
        self.device = device
        
        # Load only tokenizer and config for lightweight usage
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.config = LlamaConfig.from_pretrained(model_name)
        
        # Simple projection layer instead of full model
        self.eeg_to_text = nn.Sequential(
            nn.Linear(256, self.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.hidden_size, self.config.vocab_size)
        ).to(device)
    
    def generate_text(self, eeg_features: torch.Tensor, max_tokens: int = 16, **kwargs) -> List[str]:
        """Generate pseudo-text by projecting EEG to vocab logits and sampling top tokens."""
        logits = self.eeg_to_text(eeg_features)
        top_ids = logits.topk(k=min(max_tokens, logits.shape[-1]), dim=-1).indices
        texts = []
        for ids in top_ids:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(text)
        return texts


def create_llama_model(
    model_type: str = "adapter",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: str = "cuda",
    **kwargs
) -> nn.Module:
    """Factory function for LLaMA models
    
    Args:
        model_type: "adapter", "light"
        model_name: HuggingFace model name
        device: Device to load model on
        
    Returns:
        LLaMA model wrapper
    """
    
    if model_type == "adapter":
        return LLaMAAdapter(model_name, device, **kwargs)
    elif model_type == "light":
        return LightLLaMA(model_name, device, **kwargs)
    else:
        return LLaMAAdapter(model_name, device, **kwargs)


__all__ = ["LLaMAAdapter", "LightLLaMA", "create_llama_model"]