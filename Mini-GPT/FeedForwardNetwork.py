import torch
import torch.nn as nn
class FeedForwardNetwork(nn.Module):
    """
    Simple 2-layer MLP used after attention.
    """

    def __init__(self, embedding_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)
