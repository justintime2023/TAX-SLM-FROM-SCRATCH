import os
import pickle
import numpy as np

input_file = 'data/taxregime/input.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Length of dataset: {len(data):,} characters")

# Get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"All characters: {''.join(chars)}")

# Create mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Train/val split (90/10)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train: {len(train_ids):,} tokens")
print(f"Val: {len(val_ids):,} tokens")

# Export to binary
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join('data/taxregime', 'train.bin'))
val_ids.tofile(os.path.join('data/taxregime', 'val.bin'))

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'chars': chars,
}

with open(os.path.join('data/taxregime', 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("✓ Prepared train.bin, val.bin, and meta.pkl")
