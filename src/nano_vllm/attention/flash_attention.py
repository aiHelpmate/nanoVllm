"""FlashAttention integration for memory-efficient attention.

FlashAttention is a memory-efficient attention algorithm that:
- Fuses the attention computation into a single CUDA kernel
- Uses tiling to avoid materializing the full attention matrix (O(N) memory vs O(N^2))
- Significantly improves throughput for long sequences

This module provides a unified interface for FlashAttention, falling back to
standard attention when FlashAttention is not available.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Try to import flash_attn. The whole import block is wrapped in try/except so
# that the library remains an optional runtime dependency: on machines without
# a CUDA GPU (or without flash-attn installed) the engine still runs, just
# through the PyTorch SDPA fallback.
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_VERSION = None

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    try:
        import flash_attn
        FLASH_ATTN_VERSION = getattr(flash_attn, "__version__", "unknown")
    except Exception:
        FLASH_ATTN_VERSION = "2.x"
except ImportError:
    pass


def is_flash_attn_available() -> bool:
    """Check if FlashAttention is available."""
    return FLASH_ATTN_AVAILABLE


def get_flash_attn_version() -> Optional[str]:
    """Get FlashAttention version if available."""
    return FLASH_ATTN_VERSION if FLASH_ATTN_AVAILABLE else None


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention using FlashAttention.

    Args:
        query: [batch_size, num_heads, seq_len, head_dim]
        key: [batch_size, num_kv_heads, seq_len, head_dim]
        value: [batch_size, num_kv_heads, seq_len, head_dim]
        causal: Whether to apply causal masking
        softmax_scale: Scale factor for softmax (default: 1/sqrt(head_dim))

    Returns:
        Attention output: [batch_size, num_heads, seq_len, head_dim]
    """
    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError(
            "FlashAttention is not available. "
            "Install with: pip install flash-attn --no-build-isolation"
        )

    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    # Handle GQA: repeat KV heads if needed. FlashAttention expects the number
    # of KV heads to equal the number of query heads, so under Grouped Query
    # Attention we replicate each KV head num_groups times.
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)

    # Layout conversion: flash-attn operates on [batch, seq_len, num_heads,
    # head_dim], whereas our engine keeps tensors in [batch, num_heads,
    # seq_len, head_dim] (the same layout the model uses internally).
    query = query.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # Compute attention (fused kernel: softmax, scaling and masking all happen
    # inside the kernel, so the O(N^2) attention matrix is never materialized).
    output = flash_attn_func(
        query,
        key,
        value,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    # Convert back to our engine's [batch, num_heads, seq_len, head_dim] layout.
    output = output.transpose(1, 2)

    return output


def flash_attention_with_kv_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute attention with KV cache using FlashAttention.

    This is useful for incremental decoding where we have cached KV states.

    Args:
        query: [batch_size, num_heads, seq_len, head_dim] - new query
        key: [batch_size, num_kv_heads, seq_len, head_dim] - new key
        value: [batch_size, num_kv_heads, seq_len, head_dim] - new value
        key_cache: [batch_size, max_cache_len, num_kv_heads, head_dim]
        value_cache: [batch_size, max_cache_len, num_kv_heads, head_dim]
        cache_seqlens: [batch_size] - current cache lengths for each sequence
        causal: Whether to apply causal masking
        softmax_scale: Scale factor for softmax

    Returns:
        Tuple of (output, updated_key_cache, updated_value_cache)
    """
    if not FLASH_ATTN_AVAILABLE:
        raise RuntimeError("FlashAttention is not available")

    batch_size, num_heads, new_seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    # Handle GQA
    if num_kv_heads != num_heads:
        num_groups = num_heads // num_kv_heads
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)

    # Update cache with new KV.
    # Note: This is a simplified version; for production, use flash_attn's
    # built-in KV cache support if available (flash_attn_varlen_func with
    # cache_seqlens), which avoids gathering the full cache each step.
    for i in range(batch_size):
        cache_len = cache_seqlens[i].item()
        # key is [batch, num_heads, seq_len, head_dim] here, so transposing to
        # [seq_len, num_heads, head_dim] matches the cache's time-major layout
        # [max_cache_len, num_kv_heads, head_dim].
        key_cache[i, cache_len:cache_len + new_seq_len] = key[i].transpose(0, 1)
        value_cache[i, cache_len:cache_len + new_seq_len] = value[i].transpose(0, 1)

    # Get full KV for attention: every sequence attends over its own cached
    # history plus the newly appended tokens. Slicing to max_cache_len keeps
    # the gathered tensors tight instead of passing the whole padded cache.
    max_cache_len = cache_seqlens.max().item() + new_seq_len

    # FlashAttention expects: [batch, seq_len, num_heads, head_dim]
    query = query.transpose(1, 2)

    # Gather cached KV: cache is [batch, max_len, heads, dim]; move heads to
    # dim 1 first so the final transpose matches flash-attn's layout.
    full_key = key_cache[:, :max_cache_len].transpose(1, 2)  # [batch, num_heads, max_len, head_dim]
    full_value = value_cache[:, :max_cache_len].transpose(1, 2)

    # For variable length sequences, we need to use varlen attention.
    # For simplicity, we'll use the regular flash_attn_func with masking
    # (padding is accounted for by the causal mask only; this simplification
    # is acceptable for the engine's use case).
    full_key = full_key.transpose(1, 2)  # [batch, max_len, num_heads, head_dim]
    full_value = full_value.transpose(1, 2)

    output = flash_attn_func(
        query,
        full_key,
        full_value,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    output = output.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

    return output, key_cache, value_cache


def sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_kv_groups: int = 1,
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention via PyTorch's optimized SDPA.

    This uses torch.nn.functional.scaled_dot_product_attention which automatically
    selects the best backend:
    - FlashAttention (if available on GPU)
    - Memory-efficient attention
    - Math fallback

    Args:
        query: [batch_size, num_heads, seq_len, head_dim]
        key: [batch_size, num_kv_heads, kv_seq_len, head_dim]
        value: [batch_size, num_kv_heads, kv_seq_len, head_dim]
        attention_mask: Optional mask [batch, 1, seq_len, kv_seq_len]
        num_kv_groups: Number of query heads per KV head (for GQA)
        is_causal: Whether to apply causal masking (more efficient than explicit mask)

    Returns:
        Attention output: [batch_size, num_heads, seq_len, head_dim]
    """
    # Repeat KV heads for GQA. Using expand+reshape instead of repeat keeps the
    # tensor views (no data copy) until the reshape materializes the result.
    if num_kv_groups > 1:
        batch, num_kv_heads, kv_seq_len, head_dim = key.shape
        key = key[:, :, None, :, :].expand(
            batch, num_kv_heads, num_kv_groups, kv_seq_len, head_dim
        ).reshape(batch, num_kv_heads * num_kv_groups, kv_seq_len, head_dim)
        value = value[:, :, None, :, :].expand(
            batch, num_kv_heads, num_kv_groups, kv_seq_len, head_dim
        ).reshape(batch, num_kv_heads * num_kv_groups, kv_seq_len, head_dim)

    # PyTorch's optimized SDPA dispatches to the best available backend
    # (flash-attn, memory-efficient, or math fallback) automatically.
    # Note: passing is_causal=True is more efficient than an explicit causal
    # mask, but they are mutually exclusive — SDPA rejects both being set.
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        is_causal=is_causal and attention_mask is None,
        dropout_p=0.0,
    )

    return attn_output


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_kv_groups: int = 1,
    use_flash_attn: bool = True,
    causal: bool = True,
) -> torch.Tensor:
    """Unified attention interface that uses the best available backend.

    Priority:
    1. flash-attn library (if installed and use_flash_attn=True)
    2. PyTorch SDPA (which may use FlashAttention, memory-efficient, or math backend)

    Args:
        query: [batch_size, num_heads, seq_len, head_dim]
        key: [batch_size, num_kv_heads, kv_seq_len, head_dim]
        value: [batch_size, num_kv_heads, kv_seq_len, head_dim]
        attention_mask: Optional mask (only used for standard attention)
        num_kv_groups: Number of query heads per KV head (for GQA)
        use_flash_attn: Whether to use FlashAttention library if available
        causal: Whether to apply causal masking

    Returns:
        Attention output: [batch_size, num_heads, seq_len, head_dim]
    """
    query_len = query.shape[2]
    kv_len = key.shape[2]

    # Determine if we can use efficient is_causal mode.
    # is_causal only works when query_len == kv_len (prefill without cache).
    # For decode (query_len=1, kv_len>1), no causal mask needed because the
    # query position is the last one — everything in the cache is in its past.
    use_is_causal = causal and (query_len == kv_len) and (attention_mask is None)

    # Use flash-attn library if available and requested.
    # flash-attn handles different seq lengths properly, so GQA need not be
    # expanded here (the flash path repeats heads internally as needed).
    if use_flash_attn and FLASH_ATTN_AVAILABLE:
        return flash_attention(query, key, value, causal=causal and query_len == kv_len)

    # Fall back to PyTorch SDPA (still uses optimized backends).
    return sdpa_attention(query, key, value, attention_mask, num_kv_groups, is_causal=use_is_causal)


def print_flash_attn_info():
    """Print information about FlashAttention availability."""
    print(f"FlashAttention available: {FLASH_ATTN_AVAILABLE}")
    if FLASH_ATTN_AVAILABLE:
        print(f"FlashAttention version: {FLASH_ATTN_VERSION}")
    else:
        print("To install FlashAttention:")
        print("  pip install flash-attn --no-build-isolation")
        print("Note: Requires CUDA and compatible GPU (Ampere or newer recommended)")