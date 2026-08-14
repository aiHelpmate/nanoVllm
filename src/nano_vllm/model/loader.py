"""Load HuggingFace model weights into our Llama implementation."""

from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

from nano_vllm.config import ModelConfig
from nano_vllm.model.llama import LlamaForCausalLM


def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    use_flash_attn: bool = True,
) -> LlamaForCausalLM:
    """Load a Llama model from HuggingFace.

    Args:
        model_path: HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                   or local path to model directory
        device: Device to load model on ("cuda" or "cpu")
        dtype: Data type for model weights
        use_flash_attn: Whether to use FlashAttention if available (Phase 5)

    Returns:
        Loaded LlamaForCausalLM model
    """
    # Download model if it's a HuggingFace ID
    local_path = _get_local_path(model_path)

    # Load config
    config = ModelConfig.from_pretrained(local_path)

    print(f"Loading model: {model_path}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention heads: {config.num_attention_heads}")
    print(f"  KV heads: {config.num_key_value_heads}")
    print(f"  FlashAttention: {use_flash_attn}")

    # Create model
    model = LlamaForCausalLM(config, use_flash_attn=use_flash_attn)

    # Load weights
    state_dict = _load_weights(local_path)

    # Map HuggingFace weight names to our weight names
    mapped_state_dict = _map_weights(state_dict)

    # Load into model
    model.load_state_dict(mapped_state_dict, strict=True)

    # Move to device and dtype
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return model


def _get_local_path(model_path: str) -> Path:
    """Get local path, downloading from HuggingFace if necessary."""
    path = Path(model_path)
    if path.exists():
        # Local directory: use it directly, no hub interaction needed
        return path

    # Download from HuggingFace Hub
    # Only safetensors weights and config json are fetched; large redundant
    # files (tokenizer models, shard indexes) are skipped to keep downloads
    # lean.
    local_dir = snapshot_download(
        repo_id=model_path,
        allow_patterns=["*.safetensors", "*.json"],
    )
    return Path(local_dir)


def _load_weights(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load all weights from safetensors files."""
    state_dict = {}

    # Find all safetensors files
    # One directory may contain multiple shards (model-00001-of-00002.safetensors,
    # ...) for large models; glob collects every shard.
    safetensors_files = list(model_path.glob("*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    # Load each file
    # safetensors is loaded lazily via safe_open: only tensors we explicitly
    # request with f.get_tensor(key) are materialized, so peak memory stays
    # low even for multi-shard checkpoints.
    for filepath in safetensors_files:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    return state_dict


def _map_weights(hf_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map HuggingFace weight names to our model's weight names.

    HuggingFace Llama uses names like:
        model.embed_tokens.weight
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight
        model.layers.0.self_attn.o_proj.weight
        model.layers.0.mlp.gate_proj.weight
        model.layers.0.mlp.up_proj.weight
        model.layers.0.mlp.down_proj.weight
        model.layers.0.input_layernorm.weight
        model.layers.0.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight

    Our model uses the same names, so we just need to handle the rotary embedding
    buffers which we compute ourselves.
    """
    mapped = {}

    for name, tensor in hf_state_dict.items():
        # Skip rotary embedding buffers - we compute these ourselves
        # (RotaryEmbedding pre-computes cos/sin from inv_freq, which is why
        # the HF checkpoints don't ship these parameters for Llama anyway)
        if "rotary_emb" in name:
            continue

        # The weight names should match directly
        mapped[name] = tensor

    return mapped