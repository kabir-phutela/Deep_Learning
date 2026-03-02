import torch
import GPTModel
from Data_setup import vocab_size
from Batch_module import get_batch, block_size, device

# Model hyperparameters
embedding_dim = 256
num_heads = 4
num_layers = 4
learning_rate = 3e-4
max_iters = 5000


# Create model
model = GPTModel.GPTModel(
    vocab_size,
    embedding_dim,
    block_size,
    num_heads,
    num_layers
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print("Using device:", device)

# Training loop
for step in range(max_iters):

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "gpt_model.pth")
print("Model saved successfully.")
