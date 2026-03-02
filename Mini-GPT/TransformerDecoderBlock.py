import torch
import torch.nn as nn
import FeedForwardNetwork
import MultiHeadSelfAttention
class TransformerDecoderBlock(nn.Module):
    """
    One complete Transformer decoder block.
    Contains:
    - Multi-head attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """

    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

        self.self_attention = MultiHeadSelfAttention.MultiHeadSelfAttention(
            embedding_dim,
            num_heads,
            block_size
        )

        self.feed_forward = FeedForwardNetwork.FeedForwardNetwork(embedding_dim)

    def forward(self, x):
        # Attention with residual connection
        x = x + self.self_attention(self.layer_norm_1(x))

        # Feed forward with residual connection
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x
