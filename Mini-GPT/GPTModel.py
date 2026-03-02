import torch
import torch.nn as nn
import torch.nn.functional as F
import TransformerDecoderBlock
class GPTModel(nn.Module):
    """
    Full GPT Language Model
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        block_size,
        num_heads,
        num_layers
    ):
        super().__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional Embedding
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

        # Stack of Transformer Blocks
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerDecoderBlock.TransformerDecoderBlock(
                    embedding_dim,
                    num_heads,
                    block_size
                )
                for _ in range(num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(embedding_dim)

        # Final output layer
        self.output_linear = nn.Linear(embedding_dim, vocab_size)

        self.block_size = block_size

    def forward(self, input_indices, targets=None):
        """
        input_indices shape: (batch_size, sequence_length)
        """

        B, T = input_indices.shape

        # Token embeddings
        token_embeddings = self.token_embedding(input_indices)  # (B, T, C)

        # Position embeddings
        positions = torch.arange(T, device=input_indices.device)
        position_embeddings = self.position_embedding(positions)  # (T, C)

        # Combine
        x = token_embeddings + position_embeddings  # (B, T, C)

        # Pass through transformer blocks
        x = self.transformer_blocks(x)

        # Final normalization
        x = self.final_layer_norm(x)

        # Output logits
        logits = self.output_linear(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss
