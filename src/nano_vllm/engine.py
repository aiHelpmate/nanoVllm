"""Main inference engine with continuous batching support.

Supports two cache modes:
1. Per-sequence KV caches (Phase 2 - legacy)
2. PagedAttention with block-based KV cache (Phase 3)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict

import torch
from transformers import AutoTokenizer

from nano_vllm.config import ModelConfig
from nano_vllm.model.loader import load_model
from nano_vllm.model.llama import LlamaForCausalLM
from nano_vllm.cache import KVCache, BlockKVCache
from nano_vllm.sampler import Sampler
from nano_vllm.core.sequence import Sequence, SequenceStatus
from nano_vllm.core.scheduler import Scheduler, SchedulerOutputs, SchedulingPolicy
from nano_vllm.core.block import BLOCK_SIZE, BlockTable, compute_num_blocks
from nano_vllm.core.block_manager import BlockManager


@dataclass
class GenerationOutput:
    """Output from a completed generation."""

    seq_id: int
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int


class LLMEngine:
    """LLM inference engine with continuous batching.

    Supports processing multiple requests simultaneously with iteration-level
    scheduling. New requests can join mid-generation, and completed requests
    leave immediately.

    Key features:
    - Per-sequence KV caches (legacy) or PagedAttention
    - FCFS scheduling
    - Separate prefill and decode phases
    - Dynamic batching
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 2048,
        max_batch_size: int = 8,
        use_paged_attention: bool = True,
        num_blocks: Optional[int] = None,
        block_size: int = BLOCK_SIZE,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
        enable_preemption: bool = True,
        max_prefill_tokens: int = 512,
        enable_prefix_caching: bool = True,
        use_flash_attn: bool = True,
    ):
        """Initialize the engine.

        Args:
            model_path: HuggingFace model ID or local path
            device: Device to run on ("cuda" or "cpu")
            dtype: Model data type (float16 recommended for GPU)
            max_seq_len: Maximum sequence length for KV cache
            max_batch_size: Maximum number of sequences to process together
            use_paged_attention: If True, use PagedAttention (Phase 3)
            num_blocks: Number of blocks for PagedAttention (auto-calculated if None)
            block_size: Tokens per block for PagedAttention
            scheduling_policy: FCFS or PRIORITY scheduling
            enable_preemption: If True, can preempt low-priority for high-priority
            max_prefill_tokens: Maximum tokens to prefill per iteration (chunked prefill)
            enable_prefix_caching: If True, share KV blocks for common prefixes
            use_flash_attn: If True, use FlashAttention when available (Phase 5)
        """
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.use_paged_attention = use_paged_attention
        self.block_size = block_size
        self.scheduling_policy = scheduling_policy
        self.enable_preemption = enable_preemption
        self.max_prefill_tokens = max_prefill_tokens
        self.enable_prefix_caching = enable_prefix_caching
        self.use_flash_attn = use_flash_attn

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path, device=device, dtype=dtype, use_flash_attn=use_flash_attn)
        self.config = self.model.config
        print(f"Model loaded: {self.config.num_hidden_layers} layers, {self.config.hidden_size} hidden size")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize components
        self.sampler = Sampler()

        # Initialize PagedAttention components
        if use_paged_attention:
            # Calculate number of blocks if not provided
            if num_blocks is None:
                num_blocks = self._calculate_num_blocks()

            print(f"Using PagedAttention with {num_blocks} blocks (block_size={block_size})")

            # Create block manager and global KV cache
            self.block_manager = BlockManager(
                num_blocks=num_blocks,
                block_size=block_size,
                enable_prefix_caching=enable_prefix_caching,
            )
            self.block_kv_cache = BlockKVCache.from_config(
                config=self.config,
                num_blocks=num_blocks,
                block_size=block_size,
                device=device,
                dtype=dtype,
            )
            print(f"BlockKVCache memory: {self.block_kv_cache.memory_usage_mb:.1f}MB")

            # Scheduler with block manager
            # With a block manager attached, the scheduler can do resource-aware
            # admission (check block availability) and preemption.
            self.scheduler = Scheduler(
                max_batch_size=max_batch_size,
                block_manager=self.block_manager,
                block_size=block_size,
                scheduling_policy=scheduling_policy,
                enable_preemption=enable_preemption,
                max_prefill_tokens=max_prefill_tokens,
            )
        else:
            # Legacy mode without PagedAttention
            self.block_manager = None
            self.block_kv_cache = None
            self.scheduler = Scheduler(
                max_batch_size=max_batch_size,
                scheduling_policy=scheduling_policy,
                enable_preemption=False,  # No preemption without block manager
                max_prefill_tokens=max_prefill_tokens,
            )

        # Track prompts for output
        self._prompts: Dict[int, str] = {}

    def _calculate_num_blocks(self) -> int:
        """Calculate number of blocks based on available memory.

        For simplicity, we use a heuristic: enough blocks to support
        max_batch_size sequences each with max_seq_len tokens.
        """
        # Each sequence can need up to ceil(max_seq_len / block_size) blocks
        blocks_per_seq = (self.max_seq_len + self.block_size - 1) // self.block_size
        # Allow for max_batch_size sequences
        return blocks_per_seq * self.max_batch_size

    def add_request(
        self,
        prompt: str,
        max_tokens: int = 100,
        priority: int = 0,
    ) -> int:
        """Add a generation request to the queue.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            priority: Request priority (higher = more important)

        Returns:
            Sequence ID for tracking the request
        """
        # Tokenize prompt
        prompt_token_ids = self.tokenizer.encode(prompt)

        if len(prompt_token_ids) >= self.max_seq_len:
            raise ValueError(
                f"Prompt length {len(prompt_token_ids)} exceeds max_seq_len {self.max_seq_len}"
            )

        # Add to scheduler
        seq = self.scheduler.add_request(prompt_token_ids, max_tokens, priority=priority)

        # Store prompt for output
        self._prompts[seq.seq_id] = prompt

        return seq.seq_id

    @torch.inference_mode()
    def step(self) -> List[GenerationOutput]:
        """Run one iteration of the generation loop.

        This is the core of continuous batching:
        1. Schedule sequences for this iteration (includes preemption)
        2. Run chunked prefill for sequences in progress
        3. Run full prefill for new sequences
        4. Run decode for continuing sequences
        5. Sample next tokens
        6. Update sequence states
        7. Return any completed sequences

        Returns:
            List of completed GenerationOutput objects
        """
        # Get sequences to process
        scheduler_outputs = self.scheduler.schedule()

        if scheduler_outputs.is_empty():
            return []

        completed = []

        # Process chunked prefill sequences (partial prefill)
        # Done first so partially-prefilled sequences make progress before
        # new sequences are admitted (they are already in the running set).
        for i, seq in enumerate(scheduler_outputs.chunked_prefill_sequences):
            num_tokens = scheduler_outputs.chunked_prefill_tokens[i]
            self._run_chunked_prefill(seq, num_tokens)

        # Process full prefill sequences (new sequences, one at a time for simplicity)
        for seq in scheduler_outputs.prefill_sequences:
            self._run_prefill(seq)

        # Process decode sequences (batched)
        if scheduler_outputs.decode_sequences:
            self._run_decode(scheduler_outputs.decode_sequences)

        # Check for finished sequences
        newly_finished = self.scheduler.update_sequences(self.tokenizer.eos_token_id)

        # Convert finished sequences to outputs and free resources
        for seq in newly_finished:
            output = self._create_output(seq)
            completed.append(output)

            # Free blocks for PagedAttention
            if self.use_paged_attention and seq.block_table is not None:
                self.block_manager.free_sequence_blocks(seq.block_table)
                seq.block_table = None

        return completed

    def _run_prefill(self, seq: Sequence) -> None:
        """Run prefill phase for a new sequence.

        Processes all prompt tokens at once and initializes the KV cache.
        """
        if self.use_paged_attention:
            self._run_prefill_paged(seq)
        else:
            self._run_prefill_legacy(seq)

    def _run_prefill_legacy(self, seq: Sequence) -> None:
        """Legacy prefill with per-sequence KV cache."""
        # Create KV cache for this sequence
        seq.kv_cache = KVCache(
            config=self.config,
            max_seq_len=self.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )

        # Prepare input: all prompt tokens
        input_ids = torch.tensor(
            [seq.prompt_token_ids], dtype=torch.long, device=self.device
        )

        # Forward pass (single sequence, uses kv_cache not kv_caches)
        logits = self.model(input_ids, kv_cache=seq.kv_cache)

        # Sample first token
        next_token = self.sampler.sample(logits)
        seq.append_token(next_token.item())

    def _run_prefill_paged(self, seq: Sequence) -> None:
        """PagedAttention prefill with block-based KV cache and prefix caching."""
        prompt_len = seq.get_prompt_len()

        # Allocate blocks for the prompt (with prefix caching)
        seq.block_table, seq.shared_prefix_len = (
            self.block_manager.allocate_blocks_with_prefix_caching(seq.prompt_token_ids)
        )

        # Determine which tokens need actual computation
        # If we have cached prefix blocks, we only need to process remaining tokens
        if seq.shared_prefix_len > 0 and seq.shared_prefix_len < prompt_len:
            # Partial cache hit - process only non-cached tokens
            tokens_to_process = seq.prompt_token_ids[seq.shared_prefix_len:]
            start_pos = seq.shared_prefix_len
        elif seq.shared_prefix_len >= prompt_len:
            # Full cache hit - still need to run forward for final logits
            # but KV cache is already populated for all prompt tokens
            tokens_to_process = [seq.prompt_token_ids[-1]]  # Just last token
            start_pos = prompt_len - 1
        else:
            # No cache hit - process all tokens
            tokens_to_process = seq.prompt_token_ids
            start_pos = 0

        # Prepare input
        input_ids = torch.tensor(
            [tokens_to_process], dtype=torch.long, device=self.device
        )

        # Context length includes all prompt tokens after writing
        context_lens = [prompt_len]
        start_positions = [start_pos]

        # Forward pass with PagedAttention
        logits = self.model(
            input_ids,
            block_kv_cache=self.block_kv_cache,
            block_tables=[seq.block_table],
            context_lens=context_lens,
            start_positions=start_positions,
        )

        # Sample first token
        next_token = self.sampler.sample(logits)
        seq.append_token(next_token.item())

    def _run_chunked_prefill(self, seq: Sequence, num_tokens: int) -> None:
        """Run chunked prefill for a sequence.

        Processes a portion of the prompt tokens and updates prefill progress.
        Only samples a token when all prompt tokens have been processed.

        Args:
            seq: Sequence to process
            num_tokens: Number of tokens to prefill in this chunk
        """
        if self.use_paged_attention:
            self._run_chunked_prefill_paged(seq, num_tokens)
        else:
            # Legacy mode: just do full prefill (chunked not supported)
            self._run_prefill_legacy(seq)

    def _run_chunked_prefill_paged(self, seq: Sequence, num_tokens: int) -> None:
        """PagedAttention chunked prefill.

        Args:
            seq: Sequence to process
            num_tokens: Number of tokens to prefill in this chunk
        """
        start_pos = seq.num_prefilled_tokens
        end_pos = start_pos + num_tokens
        chunk_tokens = seq.prompt_token_ids[start_pos:end_pos]

        # Allocate blocks for the chunk if needed
        # Blocks are allocated for the length AFTER this chunk; the partial
        # final block stays uncached until it fills up via mark_block_full.
        total_tokens_after = end_pos
        blocks_needed = compute_num_blocks(total_tokens_after, self.block_size)

        if seq.block_table is None:
            seq.block_table = self.block_manager.allocate_blocks_for_sequence(blocks_needed)
        else:
            # Allocate additional blocks if needed
            current_blocks = seq.block_table.num_blocks()
            new_blocks_needed = blocks_needed - current_blocks
            for _ in range(new_blocks_needed):
                block_id = self.block_manager.allocate_block()
                seq.block_table.append_block(block_id)

        # Prepare input: chunk tokens
        input_ids = torch.tensor(
            [chunk_tokens], dtype=torch.long, device=self.device
        )

        # Context length is total tokens we'll have after this chunk
        context_lens = [end_pos]
        start_positions = [start_pos]  # Writing from this position

        # Forward pass with PagedAttention
        logits = self.model(
            input_ids,
            block_kv_cache=self.block_kv_cache,
            block_tables=[seq.block_table],
            context_lens=context_lens,
            start_positions=start_positions,
        )

        # Update prefill progress
        seq.num_prefilled_tokens = end_pos

        # If all prompt tokens are processed, sample the first output token
        if seq.num_prefilled_tokens >= len(seq.prompt_token_ids):
            next_token = self.sampler.sample(logits)
            seq.append_token(next_token.item())

    def _run_decode(self, sequences: List[Sequence]) -> None:
        """Run decode phase for continuing sequences.

        Processes one token per sequence in a batched forward pass.
        """
        if self.use_paged_attention:
            self._run_decode_paged(sequences)
        else:
            self._run_decode_legacy(sequences)

    def _run_decode_legacy(self, sequences: List[Sequence]) -> None:
        """Legacy decode with per-sequence KV caches."""
        batch_size = len(sequences)

        # Prepare batched input: last token from each sequence
        input_ids = torch.tensor(
            [[seq.get_last_token_id()] for seq in sequences],
            dtype=torch.long,
            device=self.device,
        )  # [batch_size, 1]

        # Gather KV caches
        kv_caches = [seq.kv_cache for seq in sequences]

        # Batched forward pass
        logits = self.model(input_ids, kv_caches=kv_caches)

        # Sample next tokens
        next_tokens = self.sampler.sample(logits)  # [batch_size]

        # Update each sequence
        for i, seq in enumerate(sequences):
            seq.append_token(next_tokens[i].item())

    def _run_decode_paged(self, sequences: List[Sequence]) -> None:
        """PagedAttention decode with block-based KV cache."""
        batch_size = len(sequences)

        # Maybe allocate new blocks if sequences have grown
        for seq in sequences:
            new_blocks_needed = seq.get_num_new_blocks_needed(self.block_size)
            if new_blocks_needed > 0:
                for _ in range(new_blocks_needed):
                    block_id = self.block_manager.allocate_block()
                    seq.block_table.append_block(block_id)

        # Prepare batched input: last token from each sequence
        input_ids = torch.tensor(
            [[seq.get_last_token_id()] for seq in sequences],
            dtype=torch.long,
            device=self.device,
        )  # [batch_size, 1]

        # Gather block tables and context info
        block_tables = [seq.block_table for seq in sequences]
        # Current length (before adding new token) is the start position
        # The new token will be written at cache position seq.get_len() - 1
        # (the last slot currently occupied), and after writing, context_lens
        # grows by one to include it.
        start_positions = [seq.get_len() - 1 for seq in sequences]
        # Context length is the full length including the token we're about to write
        context_lens = [seq.get_len() for seq in sequences]

        # Batched forward pass with PagedAttention
        logits = self.model(
            input_ids,
            block_kv_cache=self.block_kv_cache,
            block_tables=block_tables,
            context_lens=context_lens,
            start_positions=start_positions,
        )

        # Sample next tokens
        next_tokens = self.sampler.sample(logits)  # [batch_size]

        # Update each sequence
        for i, seq in enumerate(sequences):
            seq.append_token(next_tokens[i].item())

    def _create_output(self, seq: Sequence) -> GenerationOutput:
        """Create output from a finished sequence."""
        # Decode all tokens
        all_tokens = seq.get_token_ids()
        generated_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)

        return GenerationOutput(
            seq_id=seq.seq_id,
            prompt=self._prompts.get(seq.seq_id, ""),
            generated_text=generated_text,
            prompt_tokens=seq.get_prompt_len(),
            generated_tokens=seq.get_output_len(),
        )

    def run_to_completion(self) -> List[GenerationOutput]:
        """Run until all pending requests are complete.

        Returns:
            List of all completed GenerationOutput objects
        """
        all_outputs = []

        while self.scheduler.has_pending_requests():
            completed = self.step()
            all_outputs.extend(completed)

        # Sort by sequence ID to maintain order
        all_outputs.sort(key=lambda x: x.seq_id)
        return all_outputs

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
    ) -> str:
        """Simple single-prompt generation (backward compatible).

        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text (including prompt)
        """
        self.add_request(prompt, max_tokens)
        outputs = self.run_to_completion()
        return outputs[0].generated_text if outputs else ""

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 100,
    ) -> List[str]:
        """Generate completions for multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt

        Returns:
            List of generated texts (including prompts)
        """
        # Add all requests
        for prompt in prompts:
            self.add_request(prompt, max_tokens)

        # Run to completion
        outputs = self.run_to_completion()

        # Return just the text
        return [output.generated_text for output in outputs]

    def get_stats(self) -> dict:
        """Return engine statistics."""
        stats = {
            "model_layers": self.config.num_hidden_layers,
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "num_attention_heads": self.config.num_attention_heads,
            "num_kv_heads": self.config.num_key_value_heads,
            "max_seq_len": self.max_seq_len,
            "max_batch_size": self.max_batch_size,
            "scheduler": str(self.scheduler),
            "use_paged_attention": self.use_paged_attention,
            # Advanced scheduling (Phase 4)
            "scheduling_policy": self.scheduling_policy.value,
            "enable_preemption": self.enable_preemption,
            "max_prefill_tokens": self.max_prefill_tokens,
            "enable_prefix_caching": self.enable_prefix_caching,
            # Optimizations (Phase 5)
            "use_flash_attn": self.use_flash_attn,
            "flash_attn_available": self.model.use_flash_attn if hasattr(self.model, 'use_flash_attn') else False,
        }

        # Add PagedAttention-specific stats
        if self.use_paged_attention:
            stats.update({
                "block_size": self.block_size,
                "total_blocks": self.block_manager.num_blocks,
                "free_blocks": self.block_manager.get_num_free_blocks(),
                "used_blocks": self.block_manager.num_blocks - self.block_manager.get_num_free_blocks(),
                "kv_cache_memory_mb": self.block_kv_cache.memory_usage_mb,
            })

            # Add prefix cache stats
            if self.enable_prefix_caching:
                prefix_stats = self.block_manager.get_prefix_cache_stats()
                stats.update({
                    "prefix_cache_blocks": prefix_stats["cached_blocks"],
                    "prefix_cache_refs": prefix_stats["total_references"],
                })

        return stats