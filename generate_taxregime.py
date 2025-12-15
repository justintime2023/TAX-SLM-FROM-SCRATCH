import pickle
import torch
from model import GPTConfig, GPT

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
checkpoint = torch.load("out-taxregime/ckpt.pt", map_location=device)
config = GPTConfig(**checkpoint["model_args"])
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# Load metadata
meta = pickle.load(open("data/taxregime/meta.pkl", "rb"))
stoi, itos = meta["stoi"], meta["itos"]

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device)[None, :]

def decode(t):
    return "".join([itos[int(i)] for i in t])

# Test prompts
prompts = [
    """### TASK: NEW_TAX_REGIME
INCOME: 1210000
SALARIED: YES
### ANSWER:
""",
    """### TASK: NEW_TAX_REGIME
INCOME: 850000
SALARIED: NO
### ANSWER:
""",
    """### TASK: NEW_TAX_REGIME
INCOME: 2500000
SALARIED: YES
### ANSWER:
"""
]

print("=" * 80)
for i, prompt in enumerate(prompts, 1):
    print(f"\nTest Case {i}:")
    print("-" * 80)
    x = encode(prompt)
    
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=200,
            temperature=0.25,
            top_k=10
        )
    
    output = decode(y[0].tolist())
    print(output)
    print("=" * 80)
