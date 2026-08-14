"""Block abstractions for PagedAttention.

PagedAttention divides KV cache into fixed-size blocks, similar to
how operating systems manage virtual memory with pages.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# 用“分页”机制来管理的 KV Cache（键值缓存）显存
# Default block size (tokens per block)
BLOCK_SIZE = 16


def hash_token_block(token_ids: Tuple[int, ...], parent_hash: Optional[int] = None) -> int:
    """Compute a cumulative hash for a block of tokens including its prefix chain.

    Used for prefix caching to identify shared prefixes. The hash includes the
    parent block's hash to create a chain, ensuring that blocks at the same
    position with different prefixes have different hashes.

    Args:
        token_ids: Tuple of token IDs (must be tuple for hashability)
        parent_hash: Hash of the previous block in the chain (None for first block)

    Returns:
        Cumulative hash value that uniquely identifies this block AND all previous blocks
    """
    # Combine parent hash with current block tokens for cumulative hash.
    # This ensures blocks are only shared when the ENTIRE prefix matches:
    # two sequences share a block only if every predecessor block in their
    # prefix chain hashes identically, i.e. only when their prefixes are
    # byte-for-byte (token-for-token) equal.
    if parent_hash is None:
        return hash(token_ids)
    else:
        return hash((parent_hash, token_ids))


@dataclass
class Block:
    """A fixed-size chunk of KV cache memory.

    Each block stores KV states for `block_size` tokens.
    Blocks are the unit of allocation in the BlockManager.

    Attributes:
        block_id: Unique identifier for this physical block
        block_size: Number of tokens this block can hold
        ref_count: Reference count for prefix sharing
        prefix_hash: Hash of tokens stored in this block (for prefix caching)
        is_full: Whether the block is completely filled
    """

    block_id: int
    block_size: int = BLOCK_SIZE
    # Reference count: how many sequences currently point at this physical
    # block. When it drops to zero the block can be returned to the free pool.
    ref_count: int = 1
    # Cumulative hash of this block's tokens (see hash_token_block). Stored on
    # the block so the BlockManager can look up shared prefixes without
    # re-hashing tokens on every scheduling step.
    prefix_hash: Optional[int] = None
    is_full: bool = False

    def increment_ref(self) -> None:
        """Increment reference count (when sharing with another sequence)."""
        self.ref_count += 1

    def decrement_ref(self) -> int:
        """Decrement reference count. Returns new count."""
        self.ref_count -= 1
        return self.ref_count

    def __repr__(self) -> str:
        return f"Block(id={self.block_id}, refs={self.ref_count}, hash={self.prefix_hash})"


@dataclass
class BlockTable:
    """Maps a sequence's logical positions to physical block IDs.

    Think of this like a page table in virtual memory:
    - Logical block index: Position in the sequence (0, 1, 2, ...)
    - Physical block ID: Actual block in the global cache pool

    Token at position `p` is stored in:
    - Logical block: p // block_size
    - Slot within block: p % block_size
    - Physical block: block_ids[p // block_size]

    Example:
        block_size = 16
        sequence has 35 tokens
        block_table.block_ids = [5, 12, 3]  # 3 physical blocks

        Token 0-15  -> block 5
        Token 16-31 -> block 12
        Token 32-34 -> block 3 (slots 0-2)
    """

    block_ids: List[int] = field(default_factory=list)
    block_size: int = BLOCK_SIZE

    def get_block_id(self, logical_block_idx: int) -> int:
        """Get physical block ID for a logical block index.

        Args:
            logical_block_idx: Which block in the sequence (0-indexed)

        Returns:
            Physical block ID in the global cache
        """
        if logical_block_idx < 0:
            raise IndexError(
                f"Invalid logical block index {logical_block_idx}. "
                f"Index must be non-negative."
            )
        if logical_block_idx >= len(self.block_ids):
            max_seq_len = len(self.block_ids) * self.block_size
            requested_token_range = (
                logical_block_idx * self.block_size,
                (logical_block_idx + 1) * self.block_size - 1
            )
            raise IndexError(
                f"Logical block {logical_block_idx} not allocated. "
                f"Table has {len(self.block_ids)} blocks (max {max_seq_len} tokens). "
                f"Requested block covers tokens {requested_token_range[0]}-{requested_token_range[1]}. "
                f"Allocated blocks: {self.block_ids}"
            )
        return self.block_ids[logical_block_idx]

    def append_block(self, block_id: int) -> None:
        """Add a new physical block to the table.

        Called when the sequence grows and needs more blocks.
        """
        self.block_ids.append(block_id)

    def num_blocks(self) -> int:
        """Number of blocks allocated to this sequence."""
        return len(self.block_ids)

    def get_physical_block_ids(self) -> List[int]:
        """Get all physical block IDs for this sequence."""
        # Defensive copy: callers may reorder the table without corrupting the
        # sequence's own view of its allocation.
        return self.block_ids.copy()

    def get_slot_mapping(self, seq_len: int) -> List[int]:
        """Get physical slot indices for all tokens in the sequence.

        Returns a list where slot_mapping[i] is the global slot index
        for token i. Used for writing KV to the cache.

        Global slot = block_id * block_size + slot_within_block
        """
        slots = []
        for pos in range(seq_len):
            logical_block = pos // self.block_size
            slot_in_block = pos % self.block_size
            physical_block = self.block_ids[logical_block]
            global_slot = physical_block * self.block_size + slot_in_block
            slots.append(global_slot)
        return slots

    def __repr__(self) -> str:
        return f"BlockTable(blocks={self.block_ids})"


def compute_num_blocks(seq_len: int, block_size: int = BLOCK_SIZE) -> int:
    """Compute number of blocks needed for a sequence of given length."""
    # Ceiling division: a 35-token sequence with block_size 16 needs
    # ceil(35/16) = 3 blocks, the last one only partially filled.
    return (seq_len + block_size - 1) // block_size