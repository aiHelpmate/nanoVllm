"""PagedAttention implementation in pure PyTorch.

This is an educational implementation that demonstrates the core concepts
of PagedAttention without requiring custom CUDA kernels.
"""

import math
from typing import List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from nano_vllm.core.block import BlockTable


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: List["BlockTable"],
    context_lens: List[int],
    block_size: int,
    num_kv_heads: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention with paged KV cache.

    This function gathers K and V from non-contiguous blocks based on
    each sequence's block table, then computes standard scaled dot-product
    attention.

    Args:
        query: Query tensor [batch_size, num_heads, seq_len, head_dim]
        key_cache: Global key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Global value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: List of BlockTable, one per sequence in batch
        context_lens: List of context lengths (total tokens including new ones)
        block_size: Tokens per block
        num_kv_heads: Number of KV heads (for GQA)
        scale: Attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        Attention output [batch_size, num_heads, seq_len, head_dim]
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    device = query.device
    dtype = query.dtype

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Number of query heads per KV head (for GQA)
    num_kv_groups = num_heads // num_kv_heads

    # Find max context length for padding
    # Sequences in a batch can have different context lengths; we pad all
    # gathered tensors to the longest one and mask the padding away later.
    max_context_len = max(context_lens)

    # Gather K and V from blocks for each sequence
    # Output shape: [batch_size, num_kv_heads, max_context_len, head_dim]
    gathered_keys = torch.zeros(
        (batch_size, num_kv_heads, max_context_len, head_dim),
        device=device,
        dtype=dtype,
    )
    gathered_values = torch.zeros(
        (batch_size, num_kv_heads, max_context_len, head_dim),
        device=device,
        dtype=dtype,
    )

    # Gather from blocks for each sequence
    for batch_idx in range(batch_size):
        block_table = block_tables[batch_idx]
        context_len = context_lens[batch_idx]

        # Gather token by token from blocks
        # This is the educational version of what the real vLLM does with a
        # single fused CUDA kernel: translate each logical position into a
        # (physical_block, slot_in_block) pair via the block table, then copy.
        for pos in range(context_len):
            logical_block = pos // block_size
            slot_in_block = pos % block_size
            physical_block = block_table.block_ids[logical_block]

            # Copy from cache: [num_kv_heads, head_dim]
            gathered_keys[batch_idx, :, pos, :] = key_cache[
                physical_block, slot_in_block, :, :
            ]
            gathered_values[batch_idx, :, pos, :] = value_cache[
                physical_block, slot_in_block, :, :
            ]

    # Expand KV heads for GQA if needed
    # [batch, num_kv_heads, ctx_len, head_dim] -> [batch, num_heads, ctx_len, head_dim]
    if num_kv_groups > 1:
        gathered_keys = gathered_keys.repeat_interleave(num_kv_groups, dim=1)
        gathered_values = gathered_values.repeat_interleave(num_kv_groups, dim=1)

    # Compute attention scores: Q @ K^T
    # query: [batch, num_heads, seq_len, head_dim]
    # gathered_keys: [batch, num_heads, max_ctx_len, head_dim]
    attn_weights = torch.matmul(query, gathered_keys.transpose(-2, -1)) * scale
    # attn_weights: [batch, num_heads, seq_len, max_ctx_len]

    # Create attention mask for padding AND causal masking
    # Start with all zeros (no masking)
    attn_mask = torch.zeros(
        (batch_size, 1, seq_len, max_context_len),
        device=device,
        dtype=dtype,
    )

    # Create indices for vectorized causal mask computation
    # q_idx: [0, 1, 2, ..., seq_len-1] as column vector
    # ctx_idx: [0, 1, 2, ..., max_context_len-1] as row vector
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    ctx_idx = torch.arange(max_context_len, device=device).unsqueeze(0)

    for batch_idx in range(batch_size):
        context_len = context_lens[batch_idx]

        # Query positions start at (context_len - seq_len) in the full context
        # Example: prefill of a 10-token chunk at position 5 (context_len=10,
        # seq_len=5) means the query's local index 0 corresponds to context
        # position 5. For decode (seq_len=1), the single query is the newest
        # token at context position context_len - 1.
        start_pos = context_len - seq_len

        # Causal mask: query at position q can attend to context positions <= (start_pos + q)
        # This is: ctx_idx <= start_pos + q_idx
        # Inverted for masking: ctx_idx > start_pos + q_idx should be masked
        causal_valid = ctx_idx <= (start_pos + q_idx)

        # Also mask padding positions: ctx_idx >= context_len
        padding_valid = ctx_idx < context_len

        # Combined mask: position is valid if both causal AND not padding
        valid_mask = causal_valid & padding_valid

        # Apply mask: invalid positions get -inf
        # -inf is used because softmax maps it to exactly 0 probability,
        # which is numerically stable (unlike 0 + 1e-9 residue).
        attn_mask[batch_idx, 0, :, :] = torch.where(
            valid_mask,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float("-inf"), device=device, dtype=dtype),
        )

    attn_weights = attn_weights + attn_mask

    # Softmax
    # Computed in fp32 for numerical stability, then downcast back to the
    # model's dtype (fp16/bf16) before the final matmul.
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

    # Apply attention to values
    # attn_weights: [batch, num_heads, seq_len, max_ctx_len]
    # gathered_values: [batch, num_heads, max_ctx_len, head_dim]
    output = torch.matmul(attn_weights, gathered_values)
    # output: [batch, num_heads, seq_len, head_dim]

    return output


def write_to_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: "BlockTable",
    start_pos: int,
) -> None:
    """Write new K and V to the paged cache.

    Args:
        key: New key states [1, num_kv_heads, seq_len, head_dim]
        value: New value states [1, num_kv_heads, seq_len, head_dim]
        key_cache: Global key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Global value cache [num_blocks, block_size, num_kv_heads, head_dim]
        block_table: BlockTable for this sequence
        start_pos: Starting position in the sequence for these tokens
    """
    seq_len = key.shape[2]
    block_size = block_table.block_size

    # The exact inverse of the gather loop in paged_attention: translate each
    # logical position to a physical (block, slot) and store there. Keeping
    # both loops on the same address formula guarantees read/write symmetry.
    for i in range(seq_len):
        pos = start_pos + i
        logical_block = pos // block_size
        slot_in_block = pos % block_size
        physical_block = block_table.block_ids[logical_block]

        # Write to cache
        # key[0, :, i, :] -> key_cache[physical_block, slot_in_block, :, :]
        key_cache[physical_block, slot_in_block, :, :] = key[0, :, i, :]
        value_cache[physical_block, slot_in_block, :, :] = value[0, :, i, :]