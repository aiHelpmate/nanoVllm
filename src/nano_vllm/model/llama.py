"""Llama model implementation from scratch with batched execution support.

Supports three cache modes:
1. Single KV cache (Phase 1 - backward compat)
2. List of per-sequence KV caches (Phase 2 - continuous batching)
3. PagedAttention with block-based KV cache (Phase 3)

Optimizations (Phase 5):
- FlashAttention integration for memory-efficient attention
"""

import math
from typing import Optional, Tuple, List, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_vllm.config import ModelConfig
from nano_vllm.attention.paged_attention import paged_attention, write_to_paged_cache
from nano_vllm.attention.flash_attention import (
    is_flash_attn_available,
    attention as unified_attention,
)

if TYPE_CHECKING:
    from nano_vllm.cache import KVCache, BlockKVCache
    from nano_vllm.core.block import BlockTable


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm doesn't center the activations (no mean subtraction).
    This makes it simpler and slightly faster while maintaining effectiveness.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS: sqrt(mean(x^2))
        # (eps is added inside the sqrt for numerical stability, matching the
        # original Llama implementation)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    RoPE encodes position information by rotating the query and key vectors.
    This allows the model to learn relative positions through the dot product.
    """

    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: theta_i = base^(-2i/dim)
        # Each frequency band i rotates at a different period: low i -> fast
        # rotation (high frequency), high i -> slow rotation (low frequency).
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos and sin for all positions
        # Precomputing the full position table up to max_position_embeddings
        # avoids recomputing trig functions during every forward pass.
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cos and sin values for positions 0 to seq_len-1."""
        positions = torch.arange(seq_len).float()
        # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
        freqs = torch.outer(positions, self.inv_freq)
        # Duplicate for pairs: [seq_len, dim]
        # RoPE rotates adjacent dimension pairs (x_0, x_1), (x_2, x_3), ... so
        # the same frequency table is applied to both halves of each pair.
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given positions."""
        # x: [batch, seq_len, num_heads, head_dim]
        # position_ids: [batch, seq_len]
        cos = self.cos_cached[position_ids]  # [batch, seq_len, dim]
        sin = self.sin_cached[position_ids]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    For input [x1, x2, x3, x4], returns [-x3, -x4, x1, x2].
    This is used in the RoPE rotation formula.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    The rotation formula is:
    q_rotated = q * cos + rotate_half(q) * sin
    """
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [batch, seq_len, head_dim]
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support.

    GQA uses fewer key-value heads than query heads, with each KV head
    serving multiple query heads. This reduces memory and computation.

    Supports batched execution with per-sequence KV caches.
    Supports FlashAttention for memory-efficient computation (Phase 5).
    """

    def __init__(self, config: ModelConfig, layer_idx: int, use_flash_attn: bool = True):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # How many Q heads per KV head

        # FlashAttention support
        # Only enable if the caller asked AND the library is actually installed,
        # so the same model code runs on machines without flash-attn.
        self.use_flash_attn = use_flash_attn and is_flash_attn_available()

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional["KVCache"] = None,
        kv_caches: Optional[List["KVCache"]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # PagedAttention parameters (Phase 3)
        block_kv_cache: Optional["BlockKVCache"] = None,
        block_tables: Optional[List["BlockTable"]] = None,
        context_lens: Optional[List[int]] = None,
        start_positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Forward pass with support for single, batched, or paged KV caches.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_ids: [batch_size, seq_len]
            kv_cache: Single KV cache (for single sequence, backward compat)
            kv_caches: List of KV caches (for batched decode)
            attention_mask: Optional attention mask
            block_kv_cache: Global block-based KV cache (PagedAttention)
            block_tables: List of BlockTable (one per sequence)
            context_lens: List of context lengths (tokens cached per sequence)
            start_positions: List of start positions for writing new KV
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        # (the layout expected by our attention implementations)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # PagedAttention path (Phase 3)
        if block_kv_cache is not None and block_tables is not None:
            # Get layer-specific cache tensors
            key_cache, value_cache = block_kv_cache.get_layer_caches(self.layer_idx)

            # Write new K, V to the paged cache for each sequence
            for i in range(batch_size):
                k_i = key_states[i : i + 1]  # [1, num_kv_heads, seq_len, head_dim]
                v_i = value_states[i : i + 1]
                start_pos = start_positions[i] if start_positions else 0
                write_to_paged_cache(k_i, v_i, key_cache, value_cache, block_tables[i], start_pos)

            # Compute attention using paged cache
            attn_output = paged_attention(
                query=query_states,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                block_size=block_tables[0].block_size,
                num_kv_heads=self.num_kv_heads,
            )

            # Reshape: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.o_proj(attn_output)

        # Legacy path: Handle KV cache (Phase 1-2)
        if kv_cache is not None:
            # Single sequence mode (backward compatible)
            key_states, value_states = kv_cache.update(
                self.layer_idx, key_states, value_states
            )
        elif kv_caches is not None:
            # Batched mode: update each sequence's cache and gather
            key_states, value_states = self._update_and_gather_kv_caches(
                kv_caches, key_states, value_states
            )

        # Use unified attention (FlashAttention if available, otherwise standard)
        # causal=True: for prefill the mask is built exactly once per sequence;
        # for decode (single query) no mask is needed because the query position
        # can only see its past (handled internally by unified_attention).
        attn_output = unified_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            num_kv_groups=self.num_kv_groups,
            use_flash_attn=self.use_flash_attn,
            causal=True,
        )

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Output projection
        return self.o_proj(attn_output)

    def _update_and_gather_kv_caches(
        self,
        kv_caches: List["KVCache"],
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update each sequence's cache and gather for batched attention.

        Args:
            kv_caches: List of per-sequence KV caches
            key_states: New keys [batch, num_kv_heads, seq_len, head_dim]
            value_states: New values [batch, num_kv_heads, seq_len, head_dim]

        Returns:
            Gathered keys and values with padding
        """
        batch_size = len(kv_caches)
        # Number of new tokens in this forward step. Every sequence in the
        # batch shares the same per-step token count.
        new_seq_len = key_states.shape[2]

        # Update each cache with its new KV states
        for i, cache in enumerate(kv_caches):
            k_i = key_states[i : i + 1]  # [1, num_kv_heads, seq_len, head_dim]
            v_i = value_states[i : i + 1]
            cache.update(self.layer_idx, k_i, v_i)

        # Gather all cached states (with padding to max length)
        # Variable-length sequences are left-aligned and zero-padded to the
        # longest one in the batch; the mask built by the model hides the pad.
        #
        # IMPORTANT: the gather width must be (committed seq_len + new tokens
        # of this step), NOT cache.seq_len alone. cache.seq_len is only
        # committed in end_forward() (after the layer loop), so during the
        # forward it still holds the length from BEFORE this step; reading it
        # directly would exclude the KV just written by cache.update() above
        # and leave gather width != mask width (which uses cache.seq_len +
        # seq_len), crashing SDPA. Use the effective length instead.
        max_len = max(cache.seq_len + new_seq_len for cache in kv_caches)

        gathered_keys = torch.zeros(
            (batch_size, self.num_kv_heads, max_len, self.head_dim),
            device=key_states.device,
            dtype=key_states.dtype,
        )
        gathered_values = torch.zeros(
            (batch_size, self.num_kv_heads, max_len, self.head_dim),
            device=value_states.device,
            dtype=value_states.dtype,
        )

        for i, cache in enumerate(kv_caches):
            total_len = cache.seq_len + new_seq_len  # effective length after this step
            gathered_keys[i, :, :total_len, :] = cache.cache[self.layer_idx, 0, :, :total_len, :]
            gathered_values[i, :, :total_len, :] = cache.cache[self.layer_idx, 1, :, :total_len, :]

        return gathered_keys, gathered_values


class LlamaMLP(nn.Module):
    """SwiGLU Feed-Forward Network.

    SwiGLU uses a gating mechanism: output = down(silu(gate(x)) * up(x))
    This has been shown to work better than standard ReLU FFN.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate(x)) * up(x)
        # The gate projection decides how much of the up-projection to keep;
        # this gating mechanism outperforms plain ReLU FFNs.
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Structure: x -> norm -> attention -> + x -> norm -> mlp -> + x
    (Pre-norm architecture with residual connections)
    """

    def __init__(self, config: ModelConfig, layer_idx: int, use_flash_attn: bool = True):
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx, use_flash_attn=use_flash_attn)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional["KVCache"] = None,
        kv_caches: Optional[List["KVCache"]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # PagedAttention parameters (Phase 3)
        block_kv_cache: Optional["BlockKVCache"] = None,
        block_tables: Optional[List["BlockTable"]] = None,
        context_lens: Optional[List[int]] = None,
        start_positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids,
            kv_cache,
            kv_caches,
            attention_mask,
            block_kv_cache,
            block_tables,
            context_lens,
            start_positions,
        )
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    """Llama transformer model (without language modeling head)."""

    def __init__(self, config: ModelConfig, use_flash_attn: bool = True):
        super().__init__()
        self.config = config
        self.use_flash_attn = use_flash_attn

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx, use_flash_attn=use_flash_attn)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        kv_caches: Optional[List["KVCache"]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # PagedAttention parameters (Phase 3)
        block_kv_cache: Optional["BlockKVCache"] = None,
        block_tables: Optional[List["BlockTable"]] = None,
        context_lens: Optional[List[int]] = None,
        start_positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Create position IDs if not provided
        # The position ids must continue after any cached tokens: during decode,
        # cache.seq_len tokens are already processed, so the new tokens start at
        # that offset (this is what makes KV cache + RoPE consistent).
        if position_ids is None:
            if block_kv_cache is not None and start_positions is not None:
                # PagedAttention: use provided start positions
                # (start_positions come from the scheduler, which knows exactly
                # how many tokens each sequence has written so far)
                position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
                for i, start_pos in enumerate(start_positions):
                    position_ids[i] = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
            elif kv_cache is not None:
                # Single sequence: start from current cache position
                start_pos = kv_cache.seq_len
                position_ids = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            elif kv_caches is not None:
                # Batched: each sequence may have different position
                # (each cache tracks its own length, so offsets are per-row)
                position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=input_ids.device)
                for i, cache in enumerate(kv_caches):
                    start_pos = cache.seq_len
                    position_ids[i] = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
            else:
                # Fresh prefill: positions are simply 0..seq_len-1
                position_ids = torch.arange(seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Create causal attention mask if not provided
        # Note: For PagedAttention, the paged_attention function handles masking internally
        # Note: For SDPA with is_causal=True (prefill, no cache), we skip mask creation
        # Use float32 for mask dtype (input_ids.dtype is int64 which can't hold -inf)
        mask_dtype = torch.float32
        if attention_mask is None and block_kv_cache is None:
            if kv_cache is not None and seq_len > 1:
                # Single sequence with cache AND multiple query tokens (chunked prefill)
                # Need explicit mask because query_len != kv_len
                # (SDPA's is_causal only works when query_len == kv_len)
                attention_mask = self._create_causal_mask(
                    seq_len, input_ids.device, mask_dtype, past_len=kv_cache.seq_len
                )
                # Note: For seq_len=1 decode, no mask needed - SDPA handles it
            elif kv_caches is not None:
                # Batched with per-sequence caches - need mask for padding
                attention_mask = self._create_batched_causal_mask(
                    kv_caches, seq_len, input_ids.device, mask_dtype
                )
            # For prefill (no cache, seq_len > 1): let SDPA use is_causal=True
            # This is handled in unified_attention via causal=True parameter

        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)

        # Begin forward pass for KV caches (fixes write position for all layers)
        # All decoder layers must write at the same cache offset, so the
        # position is frozen once before the layer loop begins.
        if kv_cache is not None:
            kv_cache.begin_forward(seq_len)
        if kv_caches is not None:
            for cache in kv_caches:
                cache.begin_forward(seq_len)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_ids,
                kv_cache,
                kv_caches,
                attention_mask,
                block_kv_cache,
                block_tables,
                context_lens,
                start_positions,
            )

        # End forward pass for KV caches (commits seq_len update)
        if kv_cache is not None:
            kv_cache.end_forward()
        if kv_caches is not None:
            for cache in kv_caches:
                cache.end_forward()

        # Final normalization
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _create_causal_mask(
        self, seq_len: int, device: torch.device, dtype: torch.dtype, past_len: int = 0
    ) -> torch.Tensor:
        """Create causal attention mask.

        Returns a mask where position i can only attend to positions <= i.
        Masked positions are set to -inf, unmasked to 0.
        """
        total_len = past_len + seq_len
        # Create mask using vectorized operations (faster than for-loop)
        # Row indices: [0, 1, 2, ..., seq_len-1]
        # Col indices: [0, 1, 2, ..., total_len-1]
        # Position i can attend to columns 0 to past_len + i
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(total_len, device=device).unsqueeze(0)
        # Causal condition: col <= past_len + row
        causal_mask = col_idx <= (past_len + row_idx)
        # Convert to attention mask: True -> 0, False -> -inf
        mask = torch.where(causal_mask, torch.tensor(0.0, device=device, dtype=dtype),
                          torch.tensor(float("-inf"), device=device, dtype=dtype))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, total_len]

    def _create_batched_causal_mask(
        self,
        kv_caches: List["KVCache"],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create attention mask for batched sequences with different cache lengths.

        Each sequence can only attend to its own tokens (not other sequences).
        Padding positions are masked out.
        """
        batch_size = len(kv_caches)
        max_kv_len = max(cache.seq_len + seq_len for cache in kv_caches)

        # Start with all masked
        mask = torch.full(
            (batch_size, 1, seq_len, max_kv_len),
            float("-inf"),
            device=device,
            dtype=dtype,
        )

        # Unmask valid positions for each sequence
        for i, cache in enumerate(kv_caches):
            past_len = cache.seq_len
            total_len = past_len + seq_len
            for j in range(seq_len):
                # Position j can attend to positions 0 to past_len + j
                mask[i, 0, j, : past_len + j + 1] = 0

        return mask


class LlamaForCausalLM(nn.Module):
    """Llama model with language modeling head for text generation."""

    def __init__(self, config: ModelConfig, use_flash_attn: bool = True):
        super().__init__()
        self.config = config
        self.use_flash_attn = use_flash_attn
        self.model = LlamaModel(config, use_flash_attn=use_flash_attn)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional["KVCache"] = None,
        kv_caches: Optional[List["KVCache"]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # PagedAttention parameters (Phase 3)
        block_kv_cache: Optional["BlockKVCache"] = None,
        block_tables: Optional[List["BlockTable"]] = None,
        context_lens: Optional[List[int]] = None,
        start_positions: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Forward pass returning logits.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            kv_cache: Single KV cache (for backward compat / single sequence)
            kv_caches: List of per-sequence KV caches (for batched decode)
            attention_mask: Optional attention mask
            block_kv_cache: Global block-based KV cache (PagedAttention)
            block_tables: List of BlockTable (one per sequence)
            context_lens: List of context lengths (tokens cached per sequence)
            start_positions: List of start positions for writing new KV

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        hidden_states = self.model(
            input_ids,
            position_ids,
            kv_cache,
            kv_caches,
            attention_mask,
            block_kv_cache,
            block_tables,
            context_lens,
            start_positions,
        )
        logits = self.lm_head(hidden_states)
        return logits