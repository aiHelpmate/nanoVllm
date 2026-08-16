"""Speculative Decoding for faster inference.

Speculative decoding uses a small "draft" model to generate candidate tokens,
then verifies them in parallel with the main "target" model.

This can significantly speed up inference by generating multiple tokens per
forward pass of the large model.
"""

from nano_vllm.speculative.speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeConfig,
)

__all__ = ["SpeculativeDecoder", "SpeculativeConfig"]