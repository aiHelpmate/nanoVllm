"""Model configuration.

This module defines the single source of truth for the architectural
hyper-parameters of the Llama-style transformer we run inference with.
The fields mirror what HuggingFace stores in a model's ``config.json``,
so a model downloaded from the Hub can be instantiated with the exact
same numbers, while our engine stays decoupled from the transformers
modelling code (we never import its model classes, only this config).
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Llama models.

    A plain data holder: every field captures one architectural choice of
    the transformer. Centralising these values lets the engine, the KV cache
    and the model implementation all agree on the same shape information
    without recomputing or hard-coding anything anywhere else.
    """

    # --- Vocabulary / embeddings ------------------------------------------

    # Number of tokens in the tokenizer vocabulary. This also fixes the width
    # of the input embedding table and of the final logits projection head,
    # so every layer that touches token ids depends on it.
    vocab_size: int

    # --- Hidden dimensions -------------------------------------------------

    # Dimension of the hidden (residual) stream after embedding and inside
    # every transformer layer. All per-tensor transformations operate on
    # vectors of this width.
    # hidden_size = 每一个 token 在 Transformer 内部，用多少个数字来表示自己
    hidden_size: int

    """
        ID       向量
        0   →   [0.2, 0.1, 0.8, 0.3]   我
        1   →   [0.7, 0.4, 0.2, 0.9]   喜欢
        2   →   [0.1, 0.8, 0.3, 0.5]   吃
        3   →   [0.9, 0.2, 0.6, 0.1]   苹果
        4   →   [0.3, 0.7, 0.4, 0.8]   米饭

    - 5 个 token -> 5 行向量
    - Embedding 的大小 = [vocab_size, hidden_size] = [5, 4]
    - LM Head: 把 Transformer 输出的隐藏向量，转换成“词表中每个 token 的预测分数”
    """

    # Width of the hidden layer inside the feed-forward (MLP) block. It is
    # typically a multiple of hidden_size (e.g. 4x, as in the original Llama)
    # because the MLP projections hold most of the model's parameters.
    # MLP/FFN 内部为了进行更复杂计算而临时“扩宽”的宽度
    intermediate_size: int

    # Number of stacked transformer decoder layers. Deeper stacks capture
    # more context but cost more compute and memory per token.
    num_hidden_layers: int

    # --- Attention heads ----------------------------------------------------

    # Number of query (Q) heads. Each Q head attends over a slice of the
    # projection output of width hidden_size / num_attention_heads.
    num_attention_heads: int

    # Number of key/value (KV) heads. With Grouped Query Attention (GQA) this
    # can be smaller than num_attention_heads: several Q heads then share a
    # single KV head, which drastically cuts the size of the KV cache during
    # autoregressive generation at a small quality cost.
    num_key_value_heads: int

    """
    - Q 头数、KV 头数，决定了 head_dim = hidden_size / num_attention_heads
    """

    # Maximum sequence length the model was trained for. Rotary position
    # embeddings are pre-computed up to this length; sequences are expected
    # to stay within it.
    max_position_embeddings: int

    # --- Normalisation / positional encoding --------------------------------

    # Epsilon term added inside RMSNorm before taking the square root of the
    # variance, guarding against division by zero and numerical instability,
    # especially with low-precision (fp16/bf16) weights.
    rms_norm_eps: float

    # Base frequency of the rotary position embedding (RoPE). It controls the
    # rotation period applied to each frequency band of the position encoding;
    # the standard Llama value is 10000.0.
    rope_theta: float

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head.

        Every Q/K/V head (including shared KV heads under GQA) projects onto
        a slice of width hidden_size / num_attention_heads. This value sizes
        the attention scores matrix and the per-block KV cache storage, so it
        is derived once here instead of being recomputed in multiple modules.
        """
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load config from HuggingFace model directory or hub.

        Reads the model's ``config.json`` through transformers' AutoConfig,
        then copies the fields our engine actually needs into this lightweight
        dataclass. Routing the load through a classmethod with a lazy import
        keeps ``transformers`` an optional runtime dependency: it is only
        imported on the code path that actually talks to the Hub.

        Args:
            model_path: Local model directory or a HuggingFace hub id
                (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0").

        Returns:
            A fully populated ModelConfig matching the model's architecture.
        """
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_path)

        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, "rope_theta", 10000.0),
        )