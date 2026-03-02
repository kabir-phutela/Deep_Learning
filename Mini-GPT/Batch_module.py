from Data_setup import data
import torch

# train and test split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# hyperparamters
batch_size = 32
block_size = 128
embedding_dim = 256
num_heads = 4
num_layers = 4
learning_rate = 3e-4
max_iters = 5000
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(split):
    data_source = train_data if split == "train" else val_data

    # Random starting positions
    indices = torch.randint(
        len(data_source) - block_size,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        data_source[i:i+block_size]
        for i in indices
    ])

    # Target sequences (shifted by 1)
    y = torch.stack([
        data_source[i+1:i+block_size+1]
        for i in indices
    ])

    return x.to(device), y.to(device)

