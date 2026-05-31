# Training Example: Character-level Language Model

This example trains a small Mamba-3 model on character-level text.

```python
import torch
import torch.nn.functional as F
from mamba3_ssm import MambaLMHeadModel, MambaConfig

# ── 1. Prepare data ────────────────────────────────────
text = """
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.
""" * 100  # repeat for more data

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(t): return ''.join([itos[i] for i in t])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# ── 2. Model ───────────────────────────────────────────
cfg = MambaConfig(
    d_model=128,
    n_layer=4,
    vocab_size=vocab_size,
    ssm_cfg={
        "d_state": 64,
        "expand": 2,
        "headdim": 32,
        "is_mimo": True,
        "mimo_rank": 2,
    },
    tie_embeddings=True,
)
model = MambaLMHeadModel(cfg)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ── 3. Training loop ───────────────────────────────────
block_size = 64
batch_size = 32

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+block_size] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        l = []
        for _ in range(20):
            xb, yb = get_batch(split)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))
            l.append(loss.item())
        losses[split] = sum(l) / len(l)
    model.train()
    return losses

for step in range(2000):
    xb, yb = get_batch("train")
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 200 == 0:
        losses = estimate_loss()
        print(f"Step {step:>5} | train: {losses['train']:.4f} | val: {losses['val']:.4f}")

# ── 4. Generate text ───────────────────────────────────
model.eval()
context = torch.zeros((1, 1), dtype=torch.long)
@torch.no_grad()
def generate(model, context, max_new_tokens=200):
    for _ in range(max_new_tokens):
        logits = model(context[:, -block_size:])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat([context, next_token], dim=1)
    return context

generated = generate(model, context, 200)
print("\nGenerated:")
print(decode(generated[0].tolist()))
```

## Tips for Scaling Up

1. **Increase `d_model`** → more capacity but more memory
2. **Increase `d_state`** → larger SSM state (more memory per head)
3. **Use MIMO** (`is_mimo=True`) → better GPU utilization at decode without extra parameters
4. **Increase `n_layer`** → deeper model
5. **Add `d_intermediate`** → adds MLP after each Mamba block (like Transformer FFN)
6. **Use `ngroups > 1`** → shares B/C projections across heads to save parameters (like Grouped Query Attention)

## Comparison with Transformer

| Aspect | Transformer | Mamba-3 |
|--------|-------------|---------|
| Training | O(L²) memory | O(L) memory |
| Decode | O(L) memory per step | O(1) memory (cached state) |
| Long context | Quadratic cost | Linear cost |
| Hardware | Compute-bound at decode | Memory-bound → MIMO helps |
