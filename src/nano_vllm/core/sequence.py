"""Sequence abstraction for tracking generation requests."""

"""
通过四状态状态机（WAITING → RUNNING → SWAPPED → FINISHED）追踪请求从排队到完成的完整生命周期，封装了 prompt 输入、逐步增长的输出 token、KV cache 与 block table 的显存管理、优先级调度、分块 prefill 进度等全部状态，供调度器、引擎和模型层统一操作
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from nano_vllm.cache import KVCache
    from nano_vllm.core.block import BlockTable


class SequenceStatus(Enum):
    """Status of a sequence in the generation pipeline."""

    WAITING = "waiting"  # In queue, not yet started (no KV cache)
    RUNNING = "running"  # Currently being processed (has KV cache)
    SWAPPED = "swapped"  # Preempted, KV cache freed, waiting to resume
    FINISHED = "finished"  # Generation complete


@dataclass
class Sequence:
    """Represents a single generation request.

    Tracks the prompt tokens, generated output tokens, and generation state.
    Each sequence owns its own KV cache for efficient attention computation.
    """

    seq_id: int
    prompt_token_ids: List[int]
    max_tokens: int  # Maximum tokens to generate

    # Generated tokens (grows during generation)
    output_token_ids: List[int] = field(default_factory=list)

    # Current status (four-state machine: WAITING -> RUNNING -> SWAPPED/FINISHED)
    status: SequenceStatus = SequenceStatus.WAITING

    # KV cache for this sequence (Phase 2, legacy - initialized when status becomes RUNNING)
    kv_cache: Optional["KVCache"] = None

    # Block table for PagedAttention (Phase 3 - maps logical to physical blocks)
    block_table: Optional["BlockTable"] = None

    # Priority scheduling (Phase 4): higher value = higher priority
    priority: int = 0
    # Wall-clock arrival time: used for FIFO tie-breaking in scheduling
    arrival_time: float = field(default_factory=time.time)

    # Chunked prefill tracking (Phase 4): how many prompt tokens have been
    # processed so far. Allows a long prompt to be prefilled over multiple
    # scheduler steps instead of one giant, memory-hungry forward pass.
    num_prefilled_tokens: int = 0

    # Prefix caching (Phase 4): number of tokens using shared blocks
    shared_prefix_len: int = 0

    def get_len(self) -> int:
        """Total length of the sequence (prompt + generated tokens)."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_prompt_len(self) -> int:
        """Length of the prompt."""
        return len(self.prompt_token_ids)

    def get_output_len(self) -> int:
        """Number of tokens generated so far."""
        return len(self.output_token_ids)

    def get_token_ids(self) -> List[int]:
        """All token IDs (prompt + output)."""
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        """Get the last token ID for the next decode step.

        During prefill: returns last prompt token (but we process all)
        During decode: returns last generated token
        """
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int) -> None:
        """Append a generated token to the output."""
        self.output_token_ids.append(token_id)

    def is_finished(self, eos_token_id: int) -> bool:
        """Check if generation should stop.

        Stops if:
        - Generated EOS token
        - Reached max_tokens limit
        """
        if self.get_output_len() >= self.max_tokens:
            return True
        if self.output_token_ids and self.output_token_ids[-1] == eos_token_id:
            return True
        return False

    def is_prefill(self) -> bool:
        """Check if this sequence needs prefill (hasn't generated any tokens yet)."""
        return len(self.output_token_ids) == 0

    def is_chunked_prefill(self) -> bool:
        """Check if this sequence is in the middle of chunked prefill.

        True once some (but not all) prompt tokens have been processed:
        num_prefilled_tokens > 0 and the prompt still has unprocessed tokens.
        """
        return self.num_prefilled_tokens > 0 and self.num_prefilled_tokens < len(self.prompt_token_ids)

    def get_remaining_prefill_tokens(self) -> int:
        """Get number of tokens remaining to prefill."""
        return len(self.prompt_token_ids) - self.num_prefilled_tokens

    def get_next_chunk_tokens(self, chunk_size: int) -> List[int]:
        """Get the next chunk of tokens to prefill.

        Args:
            chunk_size: Maximum tokens in this chunk

        Returns:
            List of token IDs to process in this chunk
        """
        start = self.num_prefilled_tokens
        end = min(start + chunk_size, len(self.prompt_token_ids))
        return self.prompt_token_ids[start:end]

    def reset_for_recompute(self) -> None:
        """Reset sequence state for recomputation after preemption.

        Clears output tokens and prefill progress so the sequence
        can be re-processed from scratch.
        """
        self.output_token_ids = []
        self.num_prefilled_tokens = 0
        self.status = SequenceStatus.WAITING
        self.block_table = None
        self.kv_cache = None

    def get_num_blocks_needed(self, block_size: int) -> int:
        """Calculate how many blocks are needed for current sequence length.

        Args:
            block_size: Tokens per block

        Returns:
            Number of blocks needed (ceiling division)
        """
        return (self.get_len() + block_size - 1) // block_size

    def get_num_new_blocks_needed(self, block_size: int) -> int:
        """Calculate how many NEW blocks need to be allocated.

        This is the difference between blocks needed and blocks already allocated.
        If the sequence currently holds cached/allocated blocks (via block_table),
        only the shortfall is requested from the block manager.

        Args:
            block_size: Tokens per block

        Returns:
            Number of additional blocks needed (0 if none)
        """
        blocks_needed = self.get_num_blocks_needed(block_size)
        blocks_allocated = self.block_table.num_blocks() if self.block_table else 0
        return max(0, blocks_needed - blocks_allocated)

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, "
            f"prompt_len={self.get_prompt_len()}, "
            f"output_len={self.get_output_len()}, "
            f"status={self.status.value}, "
            f"priority={self.priority})"
        )