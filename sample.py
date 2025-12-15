import pickle
import torch
from model import GPTConfig, GPT

device = "mps" if torch.backends.mps.is_available() else "cpu"

checkpoint = torch.load("out-taxregime/ckpt.pt", map_location=device)
config = GPTConfig(**checkpoint["model_args"])
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

meta = pickle.load(open("data/taxregime/meta.pkl", "rb"))
stoi, itos = meta["stoi"], meta["itos"]

def encode(s):
    return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device)[None, :]

def decode(t):
    return "".join([itos[int(i)] for i in t])

print("Tax Regime Calculator - Interactive Mode")
print("=" * 60)

while True:
    try:
        income = input("\nEnter income (or 'quit' to exit): ")
        if income.lower() == 'quit':
            break
        
        income = int(income)
        salaried = input("Salaried? (yes/no): ").strip().lower() == 'yes'
        
        prompt = f"""### TASK: NEW_TAX_REGIME
INCOME: {income}
SALARIED: {'YES' if salaried else 'NO'}
### ANSWER:
"""
        
        x = encode(prompt)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=200, temperature=0.25, top_k=10)
        
        output = decode(y[0].tolist())
        print("\n" + "=" * 60)
        print(output)
        print("=" * 60)
        
    except ValueError:
        print("Invalid input. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting...")
        break
