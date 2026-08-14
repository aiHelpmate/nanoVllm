"""KV cache implementations for autoregressive generation.

This module provides two cache implementations:
1. KVCache: Per-sequence pre-allocated cache (Phase 2, legacy)
2. BlockKVCache: Global block-based cache for PagedAttention (Phase 3)
"""

from typing import Tuple, List

import torch

from nano_vllm.config import ModelConfig
from nano_vllm.core.block import BLOCK_SIZE


class KVCache:
    """Pre-allocated key-value cache for a single sequence.

    Each sequence owns its own KVCache instance. This enables:
    - Independent sequence lengths (no padding needed in cache)
    - Easy addition/removal of sequences in continuous batching
    - Preparation for PagedAttention in Phase 3

    Shape: [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
    The '2' dimension stores [key, value].
    """

    def __init__(
        self,
        config: ModelConfig,
        max_seq_len: int = 2048,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Current sequence length (how much of the cache is filled)
        self.seq_len = 0

        # Write position for current forward pass (saved by begin_forward)
        # This ensures all layers write to the same position
        self._write_position: int = 0
        self._pending_seq_len: int = 0

        # Pre-allocate the cache (no batch dimension - one cache per sequence)
        # Shape: [num_layers, 2, num_kv_heads, max_seq_len, head_dim]
        self.cache = torch.zeros(
            (self.num_layers, 2, self.num_kv_heads, max_seq_len, self.head_dim),
            device=device,
            dtype=dtype,
        )

    def begin_forward(self, new_seq_len: int) -> None:
        """Begin a forward pass. Must be called before update() for each layer.

        This saves the current write position so all layers write to the same location.

        Args:
            new_seq_len: Number of new tokens being processed
        """
        self._write_position = self.seq_len
        self._pending_seq_len = new_seq_len

    def end_forward(self) -> None:
        """End a forward pass. Must be called after all layers have run update().

        This commits the seq_len update after all layers have written to the cache.
        """
        self.seq_len = self._write_position + self._pending_seq_len

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key/value states and return full cached states.

        Note: Call begin_forward() before the first layer and end_forward() after
        the last layer to ensure correct write positions across all layers.

        Args:
            layer_idx: Which layer this is for
            key_states: New keys [1, num_kv_heads, seq_len, head_dim]
            value_states: New values [1, num_kv_heads, seq_len, head_dim]

        Returns:
            Tuple of (all_keys, all_values) including cached + new states
            Shape: [1, num_kv_heads, total_seq_len, head_dim]
        """
        new_seq_len = key_states.shape[2]
        write_pos = self._write_position

        # Write new states to cache at the saved write position
        # All layers use the same write_pos, ensuring correct alignment
        self.cache[layer_idx, 0, :, write_pos : write_pos + new_seq_len, :] = key_states[0]
        self.cache[layer_idx, 1, :, write_pos : write_pos + new_seq_len, :] = value_states[0]

        # Return all cached states up to and including new tokens
        end_pos = write_pos + new_seq_len
        keys = self.cache[layer_idx, 0, :, :end_pos, :].unsqueeze(0)  # Add batch dim back
        values = self.cache[layer_idx, 1, :, :end_pos, :].unsqueeze(0)

        return keys, values

    def get_seq_len(self) -> int:
        """Get current sequence length in cache."""
        return self.seq_len

    def reset(self):
        """Reset the cache for reuse."""
        self.seq_len = 0
        self._write_position = 0
        self._pending_seq_len = 0

    @property
    def memory_usage_mb(self) -> float:
        """Return memory usage in megabytes."""
        return self.cache.element_size() * self.cache.numel() / (1024 * 1024)


def gather_kv_caches(
    caches: List[KVCache],
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather KV states from multiple sequence caches for batched attention.

    For decode step, all sequences have different lengths. We gather their
    KV caches and pad to the maximum length in the batch.

    Args:
        caches: List of KVCache objects (one per sequence)
        layer_idx: Which layer to gather from

    Returns:
        keys: [batch_size, num_kv_heads, max_seq_len, head_dim]
        values: [batch_size, num_kv_heads, max_seq_len, head_dim]
    """
    if not caches:
        raise ValueError("No caches provided")

    batch_size = len(caches)
    max_len = max(cache.seq_len for cache in caches)

    # Get dimensions from first cache
    first = caches[0]
    num_kv_heads = first.num_kv_heads
    head_dim = first.head_dim
    device = first.device
    dtype = first.dtype

    # Allocate padded tensors
    keys = torch.zeros(
        (batch_size, num_kv_heads, max_len, head_dim),
        device=device,
        dtype=dtype,
    )
    values = torch.zeros(
        (batch_size, num_kv_heads, max_len, head_dim),
        device=device,
        dtype=dtype,
    )

    # Copy each sequence's cache (left-aligned, padding on right)
    for i, cache in enumerate(caches):
        seq_len = cache.seq_len
        keys[i, :, :seq_len, :] = cache.cache[layer_idx, 0, :, :seq_len, :]
        values[i, :, :seq_len, :] = cache.cache[layer_idx, 1, :, :seq_len, :]

    return keys, values


class BlockKVCache:
    """Global KV cache organized as blocks for PagedAttention.

    Unlike per-sequence caches, this is a single global pool of blocks.
    Sequences reference into this pool via their block tables.

    Shape: [num_blocks, block_size, num_kv_heads, head_dim] for each of K and V
    We store K and V separately for better memory access patterns.

    Memory layout:
        key_cache[block_id, slot, kv_head, :] = key vector
        value_cache[block_id, slot, kv_head, :] = value vector
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the block-based KV cache.

        Args:
            num_blocks: Total number of blocks in the pool
            num_layers: Number of transformer layers
            block_size: Tokens per block
            num_kv_heads: Number of KV heads
            head_dim: Dimension per head
            device: Device to allocate on
            dtype: Data type for cache
        """
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Allocate separate K and V caches for each layer
        # Shape: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.key_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self.value_cache = torch.zeros(
            (num_layers, num_blocks, block_size, num_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )

    def get_layer_caches(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get K and V caches for a specific layer.

        Returns:
            key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
            value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        """
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    @property
    def memory_usage_mb(self) -> float:
        """Total memory usage in megabytes."""
        key_bytes = self.key_cache.element_size() * self.key_cache.numel()
        value_bytes = self.value_cache.element_size() * self.value_cache.numel()
        return (key_bytes + value_bytes) / (1024 * 1024)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        num_blocks: int,
        block_size: int = BLOCK_SIZE,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "BlockKVCache":
        """Create BlockKVCache from model config.

        Args:
            config: Model configuration
            num_blocks: Number of blocks to allocate
            block_size: Tokens per block
            device: Device to allocate on
            dtype: Data type

        Returns:
            Initialized BlockKVCache
        """
        return cls(
            num_blocks=num_blocks,
            num_layers=config.num_hidden_layers,
            block_size=block_size,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            device=device,
            dtype=dtype,
        )

    def __repr__(self) -> str:
        return (
            f"BlockKVCache(blocks={self.num_blocks}, "
            f"layers={self.num_layers}, "
            f"block_size={self.block_size}, "
            f"memory={self.memory_usage_mb:.1f}MB)"
        )