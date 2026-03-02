import SelfAttentionHead
import torch
import torch.nn as nn
class MultiHeadSelfAttention(nn.Module):
    """
    Multiple attention heads working in parallel.
    """

    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()

        head_dim = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead.SelfAttentionHead(embedding_dim, head_dim, block_size)
            for _ in range(num_heads)
        ])

        # Final projection
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        # Run each head
        head_outputs = [head(x) for head in self.heads]

        # Concatenate along embedding dimension
        concatenated = torch.cat(head_outputs, dim=-1)

        # Final linear projection
        output = self.projection(concatenated)

        return output
