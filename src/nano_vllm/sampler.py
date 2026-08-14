"""Token sampling strategies for text generation.

The Sampler turns the model's raw output logits into concrete token ids.
Keeping sampling in its own module (instead of inside the engine) makes it a
single, swappable decision point: today it is greedy decoding, tomorrow it can
become temperature / top-k / top-p sampling without touching the engine.
"""

import torch

# 把模型计算出的浮点数分数（Logits），翻译成具体的离散 Token ID
class Sampler:
    """Token sampler for text generation.

    Currently implements greedy decoding (argmax).
    Future: temperature, top-k, top-p sampling.
    """

    def __init__(self):
        pass

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample next token from logits.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]

        Returns:
            next_token_ids: [batch_size] tensor of sampled token IDs
        """
        # Take logits of the last generated position only; earlier positions
        # were already sampled in previous decoding steps and are not needed.
        last_logits = logits[:, -1, :]  # [batch_size, vocab_size]

        # Greedy decoding: pick the token with the highest logit (= highest
        # probability, since the softmax is monotonic).
        next_tokens = torch.argmax(last_logits, dim=-1)  # [batch_size]

        return next_tokens