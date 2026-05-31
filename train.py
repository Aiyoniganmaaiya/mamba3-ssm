"""
train.py — Mamba-3 380M training script
======================================
RTX 4060 Laptop 8GB VRAM targeted config.

Usage:
  # Quick test with custom text
  python train.py --dataset custom --data-path myfile.txt --epochs 1

  # Train on TinyStories (auto-download)
  python train.py --dataset tinystories --epochs 3

  # Train on Wikitext-103
  python train.py --dataset wikitext --epochs 3

  # Resume from checkpoint
  python train.py --dataset tinystories --resume checkpoints/best.pt
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_ssm import MambaLMHeadModel, MambaConfig


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text=None, chars=None):
        if chars:
            self.chars = sorted(set(chars))
        elif text:
            self.chars = sorted(set(text))
        else:
            raise ValueError("Need text or chars")
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids):
        return ''.join([self.itos.get(i, '') for i in ids])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"chars": self.chars}, open(path, "w", encoding="utf-8"))

    @classmethod
    def load(cls, path):
        data = json.load(open(path, "r", encoding="utf-8"))
        return cls(chars=data["chars"])


class TextDataset:
    """Token dataset with random chunk sampling."""

    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def sample_batch(self, batch_size):
        max_start = len(self.tokens) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset too small ({len(self.tokens)} tokens) "
                f"for seq_len={self.seq_len}. Need at least {self.seq_len + 2}."
            )
        starts = [random.randint(0, max_start) for _ in range(batch_size)]
        inputs = torch.stack([
            self.tokens[s:s + self.seq_len].long() for s in starts
        ])
        targets = torch.stack([
            self.tokens[s + 1:s + 1 + self.seq_len].long() for s in starts
        ])
        return inputs, targets


def load_wikitext(data_dir="./data"):
    """Download and process wikitext-103."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "wikitext103_tokens.pt"

    if cache.exists():
        print("Loading cached wikitext-103 tokens...")
        return torch.load(cache, weights_only=True), CharTokenizer.load(data_dir / "wikitext103_tokenizer.json")

    try:
        from datasets import load_dataset
        print("Downloading wikitext-103...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        text = "\n".join(ds["train"]["text"]) + "\n".join(ds["validation"]["text"])
    except ImportError:
        raise ImportError("'datasets' not installed. Run: pip install datasets")

    tokenizer = CharTokenizer(text)
    tokenizer.save(data_dir / "wikitext103_tokenizer.json")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
    torch.save(tokens, cache)
    print(f"Wikitext-103: {len(tokens):,} tokens, vocab={tokenizer.vocab_size}")
    return tokens, tokenizer


def load_tinystories(data_dir="./data"):
    """Download and process TinyStories."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "tinystories_tokens.pt"

    if cache.exists():
        print("Loading cached TinyStories tokens...")
        return torch.load(cache, weights_only=True), CharTokenizer.load(data_dir / "tinystories_tokenizer.json")

    try:
        from datasets import load_dataset
        print("Downloading TinyStories...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        text = "\n\n".join(ds["text"][:50000])
    except ImportError:
        raise ImportError("'datasets' not installed. Run: pip install datasets")

    tokenizer = CharTokenizer(text)
    tokenizer.save(data_dir / "tinystories_tokenizer.json")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
    torch.save(tokens, cache)
    print(f"TinyStories: {len(tokens):,} tokens, vocab={tokenizer.vocab_size}")
    return tokens, tokenizer


def load_custom_file(data_path, data_dir="./data"):
    """Load a custom text file."""
    data_path = Path(data_path)
    data_dir = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    text = data_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(data_dir / f"{data_path.stem}_tokenizer.json")

    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
    print(f"Custom data: {len(tokens):,} tokens, vocab={tokenizer.vocab_size}")
    return tokens, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def cosine_warmup_lr(step, warmup_steps, total_steps, max_lr, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def build_config(args):
    return MambaConfig(
        d_model=args.d_model,
        n_layer=args.n_layer,
        vocab_size=args.vocab_size,
        ssm_cfg={
            "d_state": args.d_state,
            "expand": args.expand,
            "headdim": args.headdim,
            "is_mimo": args.is_mimo,
            "mimo_rank": args.mimo_rank,
        },
        d_intermediate=args.d_intermediate,
        rms_norm=True,
        residual_in_fp32=True,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    )


def train(args):
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")

    # Data
    print(f"\nLoading dataset: {args.dataset}")
    if args.dataset == "wikitext":
        tokens, tokenizer = load_wikitext(args.data_dir)
    elif args.dataset == "tinystories":
        tokens, tokenizer = load_tinystories(args.data_dir)
    elif args.dataset == "custom":
        tokens, tokenizer = load_custom_file(args.data_path, args.data_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    n = int(0.95 * len(tokens))
    train_ds = TextDataset(tokens[:n], args.seq_len)
    val_ds = TextDataset(tokens[n:], args.seq_len)
    args.vocab_size = tokenizer.vocab_size
    print(f"Train tokens: {len(tokens[:n]):,} | Val tokens: {len(tokens[n:]):,}")

    # Model
    config = build_config(args)
    model = MambaLMHeadModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    vram_est = total_params * 12 / 1e9
    print(f"\nModel: d_model={args.d_model} n_layer={args.n_layer} d_state={args.d_state} MIMO={args.is_mimo}")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.0f}M)")
    print(f"Est VRAM: ~{vram_est:.1f} GB")

    # Optimizer
    no_decay = {"bias", "norm", "B_bias", "C_bias", "dt_bias", "D", "B_norm", "C_norm", "norm_f"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, betas=(0.9, 0.95))

    start_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # W&B
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            use_wandb = False

    # Training loop
    model.train()
    global_step = start_step
    best_val_loss = float("inf")
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * args.epochs

    print(f"\nTraining: batch={args.batch_size}x{args.grad_accum}={effective_batch} seq_len={args.seq_len}")
    print(f"LR={args.learning_rate} warmup={args.warmup_steps} total_steps={total_steps}")
    print()

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step_in_epoch in range(steps_per_epoch):
            accum_loss = 0.0

            for _ in range(args.grad_accum):
                xb, yb = train_ds.sample_batch(args.batch_size)
                xb, yb = xb.to(device), yb.to(device)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = F.cross_entropy(logits.view(-1, model.vocab_size), yb.view(-1)) / args.grad_accum

                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # LR
            lr = cosine_warmup_lr(global_step, args.warmup_steps, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            running_loss += accum_loss

            if global_step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                ppl = math.exp(min(avg_loss, 10))
                print(f"  Step {global_step:>6} | epoch {epoch+1}/{args.epochs} | loss {avg_loss:.4f} | ppl {ppl:.1f} | lr {lr:.2e}")
                if use_wandb:
                    wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr, "step": global_step})
                running_loss = 0.0

            if global_step % args.eval_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for _ in range(args.eval_steps):
                        xb, yb = val_ds.sample_batch(args.batch_size)
                        xb, yb = xb.to(device), yb.to(device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(xb)
                            vloss = F.cross_entropy(logits.view(-1, model.vocab_size), yb.view(-1))
                        val_losses.append(vloss.item())
                val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(min(val_loss, 10))
                print(f"\n  ═══ Val: loss={val_loss:.4f} ppl={val_ppl:.1f} ═══\n")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt_path = Path(args.save_dir) / "best.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                "step": global_step, "val_loss": val_loss, "config": vars(args)}, ckpt_path)
                    print(f"  Saved best checkpoint (val_loss={val_loss:.4f})")
                model.train()

        print(f"\nEpoch {epoch+1}/{args.epochs} done in {time.time() - epoch_start:.1f}s\n")

    # Final save
    final_path = Path(args.save_dir) / "final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": global_step, "val_loss": best_val_loss, "config": vars(args)}, final_path)
    print(f"Saved: {final_path}")

    # Generate samples
    print("\n" + "=" * 60 + "\nGeneration samples\n" + "=" * 60)
    model.eval()
    for temp in [0.5, 0.8, 1.0]:
        with torch.no_grad():
            seed = torch.tensor([[random.randint(0, tokenizer.vocab_size - 1)]], device=device)
            generated = seed
            for _ in range(200):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(generated)
                    probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    generated = torch.cat([generated, next_token], dim=1)
            print(f"\n[temp={temp}]\n{tokenizer.decode(generated[0].cpu().tolist())[:200]}...")

    print(f"\nDone! Best val loss: {best_val_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba-3 on RTX 4060 Laptop 8GB")

    # Preset configs (based on actual VRAM benchmarks)
    parser.add_argument("--preset", choices=["small", "medium", "large", "380m"],
                        default=None, help="Use a benchmarked preset: small(112M), medium(306M), large(367M), 380m(original 387M)")

    # Model (defaults = medium preset)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--is-mimo", action="store_true", default=True)
    parser.add_argument("--no-mimo", action="store_true", default=False)
    parser.add_argument("--d-intermediate", type=int, default=0)

    # Data
    parser.add_argument("--dataset", choices=["wikitext", "tinystories", "custom"], default="tinystories")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--data-path", default=None)

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=256)

    # Logging
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=20)

    # Checkpoint
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--resume", default=None)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", default="mamba3")

    args = parser.parse_args()

    # Apply preset
    from mamba3_ssm.presets import CONFIGS
    if args.preset:
        p = CONFIGS[args.preset]
        args.d_model = p["d_model"]
        args.n_layer = p["n_layer"]
        args.d_state = p["d_state"]
        args.batch_size = p["batch_size"]
        args.seq_len = p["seq_len"]
        args.grad_accum = p["grad_accum"]
        print(f"Using preset '{args.preset}': {p['desc']}")

    # Fill defaults for anything not set
    if args.d_model is None:
        args.d_model = 1536
    if args.n_layer is None:
        args.n_layer = 20
    if args.batch_size is None:
        args.batch_size = 1
    if args.seq_len is None:
        args.seq_len = 256
    if args.grad_accum is None:
        args.grad_accum = 16

    if args.no_mimo:
        args.is_mimo = False
    return args


if __name__ == "__main__":
    train(parse_args())
