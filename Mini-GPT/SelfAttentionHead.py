import torch.nn as nn
import torch
import torch.nn.functional as F
class SelfAttentionHead(nn.Module):
    """
    One single attention head.
    Learns one type of relationship between tokens.
    """

    def __init__(self, embedding_dim, head_dim, block_size):
        super().__init__()

        # Linear layers to create Key, Query, Value
        self.key = nn.Linear(embedding_dim, head_dim, bias=False)
        self.query = nn.Linear(embedding_dim, head_dim, bias=False)
        self.value = nn.Linear(embedding_dim, head_dim, bias=False)

        # Causal mask (to prevent looking into future)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, embedding_dim)
        """
        B, T, C = x.shape

        # Create Key, Query, Value
        K = self.key(x)      # (B, T, head_dim)
        Q = self.query(x)    # (B, T, head_dim)
        V = self.value(x)    # (B, T, head_dim)

        # Attention scores
        attention_scores = Q @ K.transpose(-2, -1)  # (B, T, T)

        # Scale
        attention_scores = attention_scores / (C ** 0.5)

        # Apply causal mask
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:T, :T] == 0,
            float('-inf')
        )

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted sum of values
        out = attention_weights @ V  # (B, T, head_dim)

        return out
