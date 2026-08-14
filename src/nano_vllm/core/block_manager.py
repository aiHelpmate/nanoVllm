"""Block manager for PagedAttention.

Manages allocation and deallocation of KV cache blocks,
similar to how an OS manages physical memory pages.
"""

from typing import List, Optional, Dict, Tuple

from nano_vllm.core.block import (
    Block,
    BlockTable,
    BLOCK_SIZE,
    compute_block_hash,
    compute_num_blocks,
)


class BlockManager:
    """Manages allocation and deallocation of KV cache blocks.

    Maintains a pool of physical blocks and tracks which are free/used.
    Uses a simple free list (stack) for O(1) allocation/deallocation.

    Features:
    - Basic block allocation/deallocation
    - Prefix caching: share blocks for common prefixes across sequences
    - Reference counting for shared blocks

    Attributes:
        num_blocks: Total number of blocks in the pool
        block_size: Tokens per block
        free_blocks: Stack of free block IDs (LIFO for cache locality)
        enable_prefix_caching: Whether prefix caching is enabled
        prefix_cache: Maps (parent_hash, token_tuple) -> block_id for cached prefixes
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = BLOCK_SIZE,
        enable_prefix_caching: bool = True,
    ):
        """Initialize the block manager.

        Args:
            num_blocks: Total number of blocks to manage
            block_size: Number of tokens per block
            enable_prefix_caching: Whether to enable prefix caching
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        # Initialize all blocks as free (use list as stack)
        # Higher IDs at bottom, lower IDs allocated first
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))

        # Track block metadata (for ref counting)
        self.blocks: List[Block] = [
            Block(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]

        # Prefix cache: maps (parent_block_hash, token_tuple) -> block_id
        # parent_block_hash is the hash of the previous block (or None for first block)
        # This allows us to find shared prefixes across sequences
        self.prefix_cache: Dict[Tuple[Optional[int], Tuple[int, ...]], int] = {}

        # Reverse mapping: block_id -> cache key for O(1) removal
        # Without this, removing a block from prefix_cache requires O(n) search
        self._block_to_cache_key: Dict[int, Tuple[Optional[int], Tuple[int, ...]]] = {}

    def allocate_block(self) -> int:
        """Allocate a free block and return its ID.

        Returns:
            Block ID of the allocated block

        Raises:
            RuntimeError: If no free blocks available
        """
        if not self.free_blocks:
            raise RuntimeError("Out of KV cache blocks! Cannot allocate more.")

        block_id = self.free_blocks.pop()
        block = self.blocks[block_id]
        # Fresh allocation always starts with a clean state, regardless of any
        # leftover metadata from a previous life of this block.
        block.ref_count = 1
        block.prefix_hash = None
        block.is_full = False
        return block_id

    def free_block(self, block_id: int) -> None:
        """Decrement ref count and potentially free a block.

        For prefix caching, blocks are only actually freed when ref_count reaches 0.
        Shared blocks may remain in use by other sequences.

        Args:
            block_id: ID of block to free
        """
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid block ID: {block_id}")

        block = self.blocks[block_id]
        new_ref_count = block.decrement_ref()

        if new_ref_count <= 0:
            # Actually free the block: reset metadata and return to free pool.
            block.ref_count = 0
            block.prefix_hash = None
            block.is_full = False
            self.free_blocks.append(block_id)

            # Remove from prefix cache using reverse mapping (O(1) instead of O(n))
            if block_id in self._block_to_cache_key:
                cache_key = self._block_to_cache_key.pop(block_id)
                if cache_key in self.prefix_cache:
                    del self.prefix_cache[cache_key]

    def allocate_blocks_for_sequence(self, num_blocks: int) -> BlockTable:
        """Allocate multiple blocks for a new sequence (without prefix caching).

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            BlockTable with allocated block IDs

        Raises:
            RuntimeError: If not enough free blocks
        """
        if not self.can_allocate(num_blocks):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks. "
                f"Only {self.get_num_free_blocks()} available."
            )

        block_table = BlockTable(block_size=self.block_size)
        for _ in range(num_blocks):
            block_id = self.allocate_block()
            block_table.append_block(block_id)

        return block_table

    def allocate_blocks_with_prefix_caching(
        self, token_ids: List[int]
    ) -> Tuple[BlockTable, int]:
        """Allocate blocks for a sequence with prefix caching.

        Tries to find and reuse cached prefix blocks before allocating new ones.

        Args:
            token_ids: All token IDs for the sequence

        Returns:
            Tuple of (BlockTable, shared_prefix_len):
            - BlockTable with block IDs (some shared, some new)
            - Number of tokens using shared cached blocks

        Raises:
            RuntimeError: If not enough free blocks for new allocations
        """
        if not self.enable_prefix_caching:
            # Fall back to regular allocation
            num_blocks = compute_num_blocks(len(token_ids), self.block_size)
            return self.allocate_blocks_for_sequence(num_blocks), 0

        block_table = BlockTable(block_size=self.block_size)
        shared_prefix_len = 0
        parent_hash: Optional[int] = None

        # Process tokens in block-sized chunks
        num_tokens = len(token_ids)
        num_full_blocks = num_tokens // self.block_size

        # Try to find cached blocks for each full block
        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            block_tokens = tuple(token_ids[start:end])

            # Cache key chains the parent block's cumulative hash with this
            # block's own tokens, so a hit guarantees the ENTIRE prefix up to
            # and including this block is identical.
            cache_key = (parent_hash, block_tokens)

            if cache_key in self.prefix_cache:
                # Cache hit! Reuse the existing block
                cached_block_id = self.prefix_cache[cache_key]
                cached_block = self.blocks[cached_block_id]

                # Increment ref count: another sequence now points at it
                cached_block.increment_ref()
                block_table.append_block(cached_block_id)
                shared_prefix_len += self.block_size

                # Update parent hash for next block
                parent_hash = cached_block.prefix_hash
            else:
                # Cache miss - allocate new block and add to cache
                if not self.can_allocate(1):
                    raise RuntimeError("Out of KV cache blocks!")

                block_id = self.allocate_block()
                block = self.blocks[block_id]

                # Set block hash (CUMULATIVE - includes parent chain) and mark as full
                # This ensures the hash uniquely identifies the entire prefix, not just this block
                block.prefix_hash = compute_block_hash(block_tokens, parent_hash)
                block.is_full = True

                # Add to cache and reverse mapping
                self.prefix_cache[cache_key] = block_id
                self._block_to_cache_key[block_id] = cache_key

                block_table.append_block(block_id)
                parent_hash = block.prefix_hash

        # Handle the last partial block (if any)
        remaining_tokens = num_tokens % self.block_size
        if remaining_tokens > 0:
            # Partial blocks are not cached (they might grow during decode)
            if not self.can_allocate(1):
                raise RuntimeError("Out of KV cache blocks!")

            block_id = self.allocate_block()
            block_table.append_block(block_id)

        return block_table, shared_prefix_len

    def mark_block_full(
        self, block_id: int, token_ids: List[int], parent_hash: Optional[int] = None
    ) -> int:
        """Mark a block as full and add it to the prefix cache.

        Called when a sequence fills a block during generation: a block that
        was allocated as a "partial" block (kept out of the cache because it
        could still grow) graduates into a cacheable full block.

        Args:
            block_id: Block that is now full
            token_ids: Tokens stored in the block (must be exactly block_size)
            parent_hash: Cumulative hash of the previous block in the sequence
                        (includes the entire prefix chain, not just the previous block's tokens)

        Returns:
            The cumulative hash of this block (for chaining to next block)
        """
        if len(token_ids) != self.block_size:
            raise ValueError(f"Block must have exactly {self.block_size} tokens")

        if not self.enable_prefix_caching:
            return 0

        block = self.blocks[block_id]
        block_tokens = tuple(token_ids)
        # Compute CUMULATIVE hash including parent chain
        block.prefix_hash = compute_block_hash(block_tokens, parent_hash)
        block.is_full = True

        # Add to cache and reverse mapping
        cache_key = (parent_hash, block_tokens)
        self.prefix_cache[cache_key] = block_id
        self._block_to_cache_key[block_id] = cache_key

        return block.prefix_hash

    def free_sequence_blocks(self, block_table: BlockTable) -> None:
        """Free all blocks used by a sequence.

        For prefix caching, shared blocks will have their ref count decremented
        but only actually freed when no sequences reference them.

        Args:
            block_table: BlockTable containing block IDs to free
        """
        for block_id in block_table.block_ids:
            self.free_block(block_id)
        block_table.block_ids.clear()

    def get_prefix_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the prefix cache.

        Returns:
            Dictionary with cache statistics
        """
        num_cached_blocks = len(self.prefix_cache)
        total_refs = sum(
            self.blocks[bid].ref_count
            for bid in self.prefix_cache.values()
        )
        return {
            "cached_blocks": num_cached_blocks,
            "total_references": total_refs,
            "avg_refs_per_block": total_refs / num_cached_blocks if num_cached_blocks > 0 else 0,
        }

    def can_allocate(self, num_blocks: int) -> bool:
        """Check if we have enough free blocks.

        Args:
            num_blocks: Number of blocks needed

        Returns:
            True if allocation would succeed
        """
        return len(self.free_blocks) >= num_blocks

    def can_allocate_for_seq_len(self, seq_len: int) -> bool:
        """Check if we can allocate blocks for a sequence of given length.

        Args:
            seq_len: Number of tokens in the sequence

        Returns:
            True if we have enough blocks
        """
        num_blocks = compute_num_blocks(seq_len, self.block_size)
        return self.can_allocate(num_blocks)

    def get_num_free_blocks(self) -> int:
        """Number of available blocks."""
        return len(self.free_blocks)

    def get_num_used_blocks(self) -> int:
        """Number of blocks currently in use."""
        return self.num_blocks - len(self.free_blocks)

    def get_utilization(self) -> float:
        """Block utilization as a fraction (0.0 to 1.0)."""
        if self.num_blocks == 0:
            return 0.0
        return self.get_num_used_blocks() / self.num_blocks

    def __repr__(self) -> str:
        return (
            f"BlockManager(total={self.num_blocks}, "
            f"free={self.get_num_free_blocks()}, "
            f"used={self.get_num_used_blocks()})"
        )