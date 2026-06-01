"""
train.py — Mamba-3 380M training script
======================================
RTX 4060 Laptop 8GB VRAM targeted config.

Usage:
  python train.py --dataset tinystories --preset medium --epochs 3
  python train.py --dataset custom --data-path myfile.txt --epochs 1
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
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "wikitext103_tokens.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True), CharTokenizer.load(data_dir / "wikitext103_tokenizer.json")
    from datasets import load_dataset
    print("Downloading wikitext-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n".join(ds["train"]["text"]) + "\n".join(ds["validation"]["text"])
    tokenizer = CharTokenizer(text)
    tokenizer.save(data_dir / "wikitext103_tokenizer.json")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
    torch.save(tokens, cache)
    print(f"Wikitext-103: {len(tokens):,} tokens, vocab={tokenizer.vocab_size}")
    return tokens, tokenizer


def load_tinystories(data_dir="./data"):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "tinystories_tokens.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True), CharTokenizer.load(data_dir / "tinystories_tokenizer.json")
    from datasets import load_dataset
    print("Downloading TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    text = "\n\n".join(ds["text"][:50000])
    tokenizer = CharTokenizer(text)
    tokenizer.save(data_dir / "tinystories_tokenizer.json")
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.int32)
    torch.save(tokens, cache)
    print(f"TinyStories: {len(tokens):,} tokens, vocab={tokenizer.vocab_size}")
    return tokens, tokenizer


def load_custom_file(data_path, data_dir="./data"):
    data_path, data_dir = Path(data_path), Path(data_dir)
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
        d_model=args.d_model, n_layer=args.n_layer, vocab_size=args.vocab_size,
        ssm_cfg={"d_state": args.d_state, "expand": args.expand, "headdim": args.headdim,
                 "is_mimo": args.is_mimo, "mimo_rank": args.mimo_rank},
        d_intermediate=args.d_intermediate, rms_norm=True, residual_in_fp32=True,
        pad_vocab_size_multiple=8, tie_embeddings=True,
    )


def format_time(seconds):
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


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
        print(f"VRAM: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free")

    # Data
    print(f"\nLoading dataset: {args.dataset}")
    loaders = {
        "wikitext": lambda: load_wikitext(args.data_dir),
        "tinystories": lambda: load_tinystories(args.data_dir),
        "custom": lambda: load_custom_file(args.data_path, args.data_dir),
    }
    tokens, tokenizer = loaders[args.dataset]()

    n = int(0.95 * len(tokens))
    train_ds = TextDataset(tokens[:n], args.seq_len)
    val_ds = TextDataset(tokens[n:], args.seq_len)
    args.vocab_size = tokenizer.vocab_size
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = len(train_ds) // effective_batch
    total_steps = steps_per_epoch * args.epochs
    total_tokens = len(tokens[:n])

    print(f"Train: {total_tokens:,} tokens | Val: {len(tokens[n:]):,} tokens")
    print(f"Vocab: {tokenizer.vocab_size} | Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

    # Model
    config = build_config(args)
    model = MambaLMHeadModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: d_model={args.d_model} n_layer={args.n_layer} d_state={args.d_state} MIMO={args.is_mimo}")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.0f}M)")

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
    global_step = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        global_step = start_step
        print(f"Resumed from step {start_step}")

    # W&B
    use_wandb = args.wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            use_wandb = False

    # ── Training header ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Training: batch={args.batch_size} grad_accum={args.grad_accum} effective_batch={effective_batch}")
    print(f"seq_len={args.seq_len} lr={args.learning_rate} warmup={args.warmup_steps}")
    print(f"save_every={args.save_every} log_every={args.log_interval} eval_every={args.eval_interval}")
    print(f"{'='*70}\n")

    model.train()
    best_val_loss = float("inf")
    scaler = torch.amp.GradScaler("cuda")
    train_start = time.time()
    step_times = []

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_tokens = 0
        running_loss = 0.0
        optimizer.zero_grad()

        for step_in_epoch in range(steps_per_epoch):
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
            if len(step_times) > 100:
                step_times.pop(0)

            # ── Periodic logging (every log_interval steps) ────────────────
            if global_step % args.log_interval == 0:
                elapsed = time.time() - train_start
                avg_step_time = sum(step_times) / len(step_times) if step_times else 0
                tokens_per_sec = epoch_tokens / max(time.time() - epoch_start, 1e-6)
                avg_loss = running_loss / args.log_interval
                ppl = math.exp(min(avg_loss, 10))

                # ETA estimation
                steps_done = global_step - start_step
                steps_remaining = total_steps - steps_done
                eta_seconds = avg_step_time * steps_remaining if avg_step_time > 0 else 0
                progress_pct = steps_done / max(total_steps, 1) * 100

                print(
                    f"  [{global_step:>6}/{total_steps}] "
                    f"epoch={epoch+1}/{args.epochs} "
                    f"loss={avg_loss:.4f} ppl={ppl:.1f} "
                    f"lr={lr:.2e} "
                    f"tok/s={tokens_per_sec:.0f} "
                    f"progress={progress_pct:.1f}% "
                    f"elapsed={format_time(elapsed)} "
                    f"eta={format_time(eta_seconds)}"
                )

                if use_wandb:
                    wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "train/lr": lr,
                               "train/tok_per_sec": tokens_per_sec, "step": global_step})
                running_loss = 0.0

            # ── Save checkpoint (every save_every steps) ──────────────────
            if args.save_every and global_step % args.save_every == 0:
                ckpt_path = Path(args.save_dir) / f"step_{global_step}.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "val_loss": best_val_loss,
                    "train_loss": running_loss,
                    "config": vars(args),
                }, ckpt_path)
                print(f"  >> Saved checkpoint: {ckpt_path.name}")

            # ── Validation (every eval_interval steps) ────────────────────
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

                print(f"\n  {'═'*60}")
                print(f"  Validation @ step {global_step}: loss={val_loss:.4f} ppl={val_ppl:.1f}")
                print(f"  {'═'*60}\n")

                if use_wandb:
                    wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "step": global_step})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = Path(args.save_dir) / "best.pt"
                    torch.save({
                        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": global_step, "val_loss": val_loss, "config": vars(args),
                    }, best_path)
                    print(f"  >> New best! Saved: {best_path}\n")
                model.train()

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{args.epochs} done in {format_time(epoch_time)} "
              f"({epoch_tokens/epoch_time:.0f} tok/s avg)\n")

    # ── Final save ─────────────────────────────────────────────────────────
    final_path = Path(args.save_dir) / "final.pt"
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "step": global_step, "val_loss": best_val_loss, "config": vars(args),
    }, final_path)

    total_time = time.time() - train_start
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Steps: {global_step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final checkpoint: {final_path}")
    print(f"{'═'*70}\n")

    # ── Generate samples ──────────────────────────────────────────────────
    print("Generation samples\n" + "="*60)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba-3 on RTX 4060 Laptop 8GB")

    # Presets
    parser.add_argument("--preset", choices=["small", "medium", "large"],
                        default=None, help="small(112M,bs2,sl512) medium(306M,bs1,sl256) large(367M,bs1,sl256)")

    # Model
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

    # Logging & checkpointing
    parser.add_argument("--log-interval", type=int, default=50, help="Print loss every N steps")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N steps (0=disabled)")
    parser.add_argument("--eval-interval", type=int, default=500, help="Run validation every N steps (0=disabled)")
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

    # Defaults
    if args.d_model is None: args.d_model = 1536
    if args.n_layer is None: args.n_layer = 20
    if args.batch_size is None: args.batch_size = 1
    if args.seq_len is None: args.seq_len = 256
    if args.grad_accum is None: args.grad_accum = 16

    if args.no_mimo:
        args.is_mimo = False
    return args


if __name__ == "__main__":
    train(parse_args())
