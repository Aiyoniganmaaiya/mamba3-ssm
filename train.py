"""
train.py — Mamba-3 training script
=================================
RTX 4060 Laptop 8GB VRAM targeted. All checkpoints support resume.

Usage:
  python train.py --dataset tinystories --preset medium --epochs 3
  python train.py --dataset custom --data-path myfile.txt --preset small --epochs 1
  python train.py --dataset tinystories --resume checkpoints/best.pt
  python train.py --dataset tinystories --resume checkpoints/step_5000.pt
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
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(path, model, optimizer, scaler, global_step, best_val_loss,
                    train_loss=0.0, val_loss=float("inf"), config=None):
    """Save a complete, resumable checkpoint."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": global_step,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "train_loss": train_loss,
        "config": config or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    """Load a checkpoint. Returns (step, best_val_loss)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["step"], ckpt.get("best_val_loss", float("inf"))


def find_latest_checkpoint(save_dir):
    """Find the most recent step_N.pt checkpoint, or best.pt, or None."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    # Prefer step_N.pt (sorted by step number)
    step_ckpts = sorted(save_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]), reverse=True)
    if step_ckpts:
        return str(step_ckpts[0])
    # Fallback to best.pt
    best = save_dir / "best.pt"
    if best.exists():
        return str(best)
    final = save_dir / "final.pt"
    if final.exists():
        return str(final)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
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

    def encode(self, s): return [self.stoi.get(c, 0) for c in s]
    def decode(self, ids): return ''.join([self.itos.get(i, '') for i in ids])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"chars": self.chars}, open(path, "w", encoding="utf-8"))

    @classmethod
    def load(cls, path):
        data = json.load(open(path, "r", encoding="utf-8"))
        return cls(chars=data["chars"])


class TextDataset:
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def sample_batch(self, batch_size):
        max_start = len(self.tokens) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(f"Dataset too small for seq_len={self.seq_len}")
        starts = [random.randint(0, max_start) for _ in range(batch_size)]
        return (
            torch.stack([self.tokens[s:s+self.seq_len].long() for s in starts]),
            torch.stack([self.tokens[s+1:s+1+self.seq_len].long() for s in starts]),
        )


def load_wikitext(data_dir="./data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "wikitext103_tokens.pt"
    tok_path = data_dir / "wikitext103_tokenizer.json"
    if cache.exists():
        return torch.load(cache, weights_only=True), CharTokenizer.load(tok_path)
    from datasets import load_dataset
    print("Downloading wikitext-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n".join(ds["train"]["text"]) + "\n".join(ds["validation"]["text"])
    tok = CharTokenizer(text); tok.save(tok_path)
    tokens = torch.tensor(tok.encode(text), dtype=torch.int32)
    torch.save(tokens, cache)
    print(f"Wikitext-103: {len(tokens):,} tokens, vocab={tok.vocab_size}")
    return tokens, tok


def load_tinystories(data_dir="./data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "tinystories_tokens.pt"
    tok_path = data_dir / "tinystories_tokenizer.json"
    if cache.exists():
        return torch.load(cache, weights_only=True), CharTokenizer.load(tok_path)
    from datasets import load_dataset
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    text = "\n\n".join(ds["text"][:50000])
    tok = CharTokenizer(text); tok.save(tok_path)
    tokens = torch.tensor(tok.encode(text), dtype=torch.int32)
    torch.save(tokens, cache)
    print(f"TinyStories: {len(tokens):,} tokens, vocab={tok.vocab_size}")
    return tokens, tok


def load_custom_file(data_path, data_dir="./data"):
    data_path, data_dir = Path(data_path), Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Not found: {data_path}")
    text = data_path.read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data_dir.mkdir(parents=True, exist_ok=True)
    tok.save(data_dir / f"{data_path.stem}_tokenizer.json")
    tokens = torch.tensor(tok.encode(text), dtype=torch.int32)
    print(f"Custom: {len(tokens):,} tokens, vocab={tok.vocab_size}")
    return tokens, tok


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def cosine_warmup_lr(step, warmup, total, max_lr, min_lr=1e-5):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


def fmt_time(s):
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free")

    # Data
    print(f"\nLoading: {args.dataset}")
    tokens, tokenizer = {
        "wikitext": lambda: load_wikitext(args.data_dir),
        "tinystories": lambda: load_tinystories(args.data_dir),
        "custom": lambda: load_custom_file(args.data_path, args.data_dir),
    }[args.dataset]()

    n = int(0.95 * len(tokens))
    train_tokens = len(tokens[:n])
    train_ds, val_ds = TextDataset(tokens[:n], args.seq_len), TextDataset(tokens[n:], args.seq_len)
    args.vocab_size = tokenizer.vocab_size
    eff_batch = args.batch_size * args.grad_accum
    # Each step consumes seq_len * eff_batch tokens
    tokens_per_step = args.seq_len * eff_batch
    steps_per_epoch = train_tokens // tokens_per_step
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs

    print(f"Train: {train_tokens:,} | Val: {len(tokens[n:]):,} | Vocab: {tokenizer.vocab_size}")
    print(f"Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Eff batch: {eff_batch} | Tokens/step: {tokens_per_step}")

    # Model
    config = MambaConfig(
        d_model=args.d_model, n_layer=args.n_layer, vocab_size=args.vocab_size,
        ssm_cfg={"d_state": args.d_state, "expand": args.expand, "headdim": args.headdim,
                 "is_mimo": args.is_mimo, "mimo_rank": args.mimo_rank},
        d_intermediate=args.d_intermediate, rms_norm=True, residual_in_fp32=True,
        pad_vocab_size_multiple=8, tie_embeddings=True,
    )
    model = MambaLMHeadModel(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: d={args.d_model} L={args.n_layer} D={args.d_state} MIMO={args.is_mimo}")
    print(f"Parameters: {params:,} ({params/1e6:.0f}M)")

    # Optimizer
    no_decay = {"bias", "norm", "B_bias", "C_bias", "dt_bias", "D", "B_norm", "C_norm", "norm_f"}
    pgs = [
        {"params": [p for n, p in model.named_parameters() if not any(d in n for d in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(d in n for d in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(pgs, lr=args.learning_rate, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")

    # Resume
    global_step = 0
    best_val_loss = float("inf")
    if args.resume:
        if not Path(args.resume).exists():
            # Try auto-finding latest checkpoint
            args.resume = find_latest_checkpoint(args.save_dir)
        if args.resume and Path(args.resume).exists():
            global_step, best_val_loss = load_checkpoint(args.resume, model, optimizer, scaler, device)
            print(f"Resumed from {args.resume} (step={global_step}, best_val={best_val_loss:.4f})")
        else:
            print(f"Warning: no checkpoint found, starting from scratch")

    # W&B
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            use_wandb = False

    # ── Training header ───────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"Training: batch={args.batch_size}×{args.grad_accum}={eff_batch} | seq_len={args.seq_len}")
    print(f"lr={args.learning_rate} | warmup={args.warmup_steps} | total={total_steps} steps")
    print(f"save_every={args.save_every} | log_every={args.log_interval} | eval_every={args.eval_interval}")
    print(f"{'═'*70}\n")

    model.train()
    train_start = time.time()
    step_times = []
    history = []  # (step, train_loss, val_loss) for summary

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_tokens = 0
        running_loss = 0.0
        optimizer.zero_grad()

        for _ in range(steps_per_epoch):
            step_t0 = time.time()
            accum_loss = 0.0

            for _ in range(args.grad_accum):
                xb, yb = train_ds.sample_batch(args.batch_size)
                xb, yb = xb.to(device), yb.to(device)
                epoch_tokens += xb.numel()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = F.cross_entropy(logits.view(-1, model.vocab_size), yb.view(-1)) / args.grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            lr = cosine_warmup_lr(global_step, args.warmup_steps, total_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            running_loss += accum_loss
            step_times.append(time.time() - step_t0)
            if len(step_times) > 200:
                step_times = step_times[-100:]

            # ── Log every N steps ────────────────────────────────────────
            if global_step % args.log_interval == 0:
                elapsed = time.time() - train_start
                avg_step = sum(step_times) / len(step_times) if step_times else 0
                tok_s = epoch_tokens / max(time.time() - epoch_start, 1e-6)
                avg_loss = running_loss / args.log_interval
                ppl = math.exp(min(avg_loss, 10))
                steps_done = global_step
                eta = avg_step * (total_steps - steps_done) if avg_step > 0 else 0
                pct = steps_done / max(total_steps, 1) * 100

                print(
                    f"  [{global_step:>6}/{total_steps}] "
                    f"ep={epoch+1}/{args.epochs} "
                    f"loss={avg_loss:.4f} ppl={ppl:.1f} "
                    f"lr={lr:.2e} "
                    f"tok/s={tok_s:.0f} "
                    f"progress={pct:.1f}% "
                    f"elapsed={fmt_time(elapsed)} "
                    f"eta={fmt_time(eta)}"
                )
                if use_wandb:
                    wandb.log({"train/loss": avg_loss, "train/ppl": ppl,
                               "train/lr": lr, "train/tok_per_sec": tok_s, "step": global_step})
                running_loss = 0.0

            # ── Save checkpoint every N steps ────────────────────────────
            if args.save_every and global_step % args.save_every == 0:
                path = Path(args.save_dir) / f"step_{global_step}.pt"
                save_checkpoint(path, model, optimizer, scaler, global_step,
                                best_val_loss, train_loss=running_loss, config=vars(args))
                print(f"  ✓ Checkpoint saved: {path.name}")

            # ── Validate every N steps ───────────────────────────────────
            if args.eval_interval and global_step % args.eval_interval == 0:
                model.eval()
                vl = []
                with torch.no_grad():
                    for _ in range(args.eval_steps):
                        xb, yb = val_ds.sample_batch(args.batch_size)
                        xb, yb = xb.to(device), yb.to(device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            vl.append(F.cross_entropy(model(xb).view(-1, model.vocab_size), yb.view(-1)).item())
                val_loss = sum(vl) / len(vl)
                val_ppl = math.exp(min(val_loss, 10))
                history.append((global_step, running_loss, val_loss))

                print(f"\n  {'═'*60}")
                print(f"  Validation @ step {global_step}: loss={val_loss:.4f}  ppl={val_ppl:.1f}")
                print(f"  {'═'*60}\n")

                if use_wandb:
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "step": global_step})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(Path(args.save_dir) / "best.pt", model, optimizer, scaler,
                                    global_step, best_val_loss, val_loss=val_loss, config=vars(args))
                    print(f"  ★ New best! val_loss={val_loss:.4f}\n")
                model.train()

        print(f"\nEpoch {epoch+1}/{args.epochs} done in {fmt_time(time.time()-epoch_start)} "
              f"({epoch_tokens/max(time.time()-epoch_start,1):.0f} tok/s)\n")

    # ── Final save ─────────────────────────────────────────────────────────
    save_checkpoint(Path(args.save_dir) / "final.pt", model, optimizer, scaler,
                    global_step, best_val_loss, config=vars(args))

    total_time = time.time() - train_start
    print(f"\n{'═'*70}")
    print(f"Training complete!")
    print(f"  Total time:  {fmt_time(total_time)}")
    print(f"  Steps:       {global_step}")
    print(f"  Best val:    {best_val_loss:.4f}")
    print(f"  Final ckpt:  checkpoints/final.pt")
    print(f"  Best ckpt:   checkpoints/best.pt")
    if history:
        print(f"  Val history: {len(history)} evaluations")
    print(f"{'═'*70}\n")

    # ── Generate samples ──────────────────────────────────────────────────
    print("Generation samples\n" + "="*60)
    model.eval()
    for temp in [0.5, 0.8, 1.0]:
        with torch.no_grad():
            seed = torch.tensor([[random.randint(0, tokenizer.vocab_size-1)]], device=device)
            gen = seed
            for _ in range(200):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(gen)
                    probs = F.softmax(logits[:, -1, :] / temp, dim=-1)
                    gen = torch.cat([gen, torch.multinomial(probs, 1)], dim=1)
            print(f"\n[temp={temp}]\n{tokenizer.decode(gen[0].cpu().tolist())[:200]}...")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train Mamba-3")
    p.add_argument("--preset", choices=["small", "medium", "large"], default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--headdim", type=int, default=64)
    p.add_argument("--mimo-rank", type=int, default=4)
    p.add_argument("--is-mimo", action="store_true", default=True)
    p.add_argument("--no-mimo", action="store_true", default=False)
    p.add_argument("--d-intermediate", type=int, default=0)
    p.add_argument("--dataset", choices=["wikitext", "tinystories", "custom"], default="tinystories")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--data-path", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Override total steps (0 = use epochs). Useful for quick experiments.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint, or auto-find latest in save-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="mamba3")
    args = p.parse_args()

    from mamba3_ssm.presets import CONFIGS
    if args.preset:
        conf = CONFIGS[args.preset]
        args.d_model = conf["d_model"]
        args.n_layer = conf["n_layer"]
        args.d_state = conf["d_state"]
        args.batch_size = conf["batch_size"]
        args.seq_len = conf["seq_len"]
        args.grad_accum = conf["grad_accum"]
        print(f"Preset '{args.preset}': {conf['desc']}")

    if args.d_model is None: args.d_model = 1536
    if args.n_layer is None: args.n_layer = 20
    if args.batch_size is None: args.batch_size = 1
    if args.seq_len is None: args.seq_len = 256
    if args.grad_accum is None: args.grad_accum = 16
    if args.no_mimo: args.is_mimo = False
    return args


if __name__ == "__main__":
    train(parse_args())
