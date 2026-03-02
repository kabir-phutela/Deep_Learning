import torch
import torch.nn.functional as F

import GPTModel
from Data_setup import vocab_size, encode, decode
from Batch_module import block_size, device


# -------------------------
# Model Hyperparameters
# (must match train.py)
# -------------------------
embedding_dim = 256
num_heads = 4
num_layers = 4


# -------------------------
# Create Model Instance
# -------------------------
model = GPTModel.GPTModel(
    vocab_size,
    embedding_dim,
    block_size,
    num_heads,
    num_layers
).to(device)

# -------------------------
# Load Trained Weights
# -------------------------
model.load_state_dict(
    torch.load("gpt_model.pth", map_location=device)
)

model.eval()


# -------------------------
# Generate Function
# -------------------------
def generate(start_text, max_new_tokens=200, temperature=1.0):

    input_ids = torch.tensor(
        encode(start_text),
        dtype=torch.long
    ).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):

        # Crop to block size
        input_crop = input_ids[:, -block_size:]

        logits, _ = model(input_crop)

        # Take last token logits
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return decode(input_ids[0].tolist())


# -------------------------
# Test Generation
# -------------------------
if __name__ == "__main__":

    prompt = "To be, or not to be"

    output_text = generate(prompt, max_new_tokens=300)

    print("\nGenerated Text:\n")
    print(output_text)
