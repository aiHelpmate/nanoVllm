"""Tests for Phase 5 optimizations: FlashAttention."""

import pytest
import torch

from nano_vllm.attention.flash_attention import (
    is_flash_attn_available,
    get_flash_attn_version,
    sdpa_attention,
    attention,
    print_flash_attn_info,
)


class TestFlashAttention:
    """Tests for FlashAttention integration."""

    def test_flash_attn_availability_check(self):
        """Test that availability check works."""
        # Should not raise
        is_available = is_flash_attn_available()
        assert isinstance(is_available, bool)

    def test_flash_attn_version(self):
        """Test version retrieval."""
        version = get_flash_attn_version()
        if is_flash_attn_available():
            assert version is not None
            assert isinstance(version, str)
        else:
            assert version is None

    def test_sdpa_attention_basic(self):
        """Test SDPA attention computation."""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 32

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output = sdpa_attention(query, key, value)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_sdpa_attention_with_mask(self):
        """Test SDPA attention with causal mask."""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 32

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Create causal mask
        mask = torch.full((seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = sdpa_attention(query, key, value, attention_mask=mask)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_sdpa_attention_gqa(self):
        """Test SDPA attention with grouped query attention."""
        batch_size = 2
        num_heads = 8
        num_kv_heads = 2
        num_kv_groups = num_heads // num_kv_heads
        seq_len = 8
        head_dim = 32

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

        output = sdpa_attention(
            query, key, value, num_kv_groups=num_kv_groups
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_unified_attention_interface(self):
        """Test unified attention interface."""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 32

        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Should work regardless of FlashAttention availability
        output = attention(
            query, key, value,
            use_flash_attn=False,  # Force SDPA path
            causal=True,
        )

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_print_flash_attn_info(self, capsys):
        """Test info printing function."""
        print_flash_attn_info()
        captured = capsys.readouterr()
        assert "FlashAttention available:" in captured.out


class TestAttentionNumerics:
    """Numerical tests for attention implementations."""

    def test_attention_values_reasonable(self):
        """Test that attention output values are in reasonable range."""
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 32

        # Use normalized inputs
        query = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1
        key = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1
        value = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1

        output = sdpa_attention(query, key, value)

        # Output should be roughly same scale as value (due to softmax normalization)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_causal_mask_prevents_future_attention(self):
        """Test that causal mask prevents attending to future positions."""
        batch_size = 1
        num_heads = 1
        seq_len = 4
        head_dim = 8

        # Create inputs where each position is distinct
        # value is an identity-like matrix [seq_len, head_dim] where position i
        # has a 1 at column i: attending only to past positions yields a
        # weighted combination of past identity rows.
        query = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        key = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        eye = torch.eye(seq_len)
        value = torch.zeros(batch_size, num_heads, seq_len, head_dim)
        value[0, 0, :, :seq_len] = eye

        # Create causal mask
        mask = torch.full((seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        output = sdpa_attention(query, key, value, attention_mask=mask)

        # All-zero queries+keys give uniform softmax over unmasked positions:
        # position i's output is the average of value rows 0..i. Future rows
        # must contribute nothing (causal), so columns > i stay zero.
        row = output[0, 0, 2, :seq_len]  # e.g. position 2 attends to rows 0..2
        assert torch.allclose(row[:3], torch.full((3,), 1 / 3)), row
        assert row[3] == 0, row  # future position 3 must not leak in


class TestIntegration:
    """Integration tests for optimization features."""

    def test_attention_dtype_preservation(self):
        """Test that attention preserves input dtype."""
        for dtype in [torch.float32, torch.float16]:
            batch_size = 2
            num_heads = 4
            seq_len = 8
            head_dim = 32

            query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
            key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
            value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

            output = sdpa_attention(query, key, value)

            assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for GPU tests"
    )
    def test_attention_gpu(self):
        """Test attention on GPU."""
        batch_size = 2
        num_heads = 4
        seq_len = 8
        head_dim = 32

        query = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
        value = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")

        output = sdpa_attention(query, key, value)

        assert output.device.type == "cuda"
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])