import torch
import torch.nn as nn
import torch.nn.functional as F

# Load dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
