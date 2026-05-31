"""
generate.py — Text generation with a trained Mamba-3 checkpoint.

Usage:
  python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" --max-tokens 200
"""

import argparse
import json
import random
import torch
import torch.nn.functional as F
from pathlib import Path

from mamba3_ssm import MambaLMHeadModel, MambaConfig


def load_checkpoint(ckpt_path):
    """Load model and tokenizer from checkpoint directory."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_args = ckpt["config"]

    # Rebuild config
    ssm_cfg = {
        "d_state": cfg_args.get("d_state", 128),
        "expand": cfg_args.get("expand", 2),
        "headdim": cfg_args.get("headdim", 64),
        "is_mimo": cfg_args.get("is_mimo", True),
        "mimo_rank": cfg_args.get("mimo_rank", 4),
    }
    config = MambaConfig(
        d_model=cfg_args["d_model"],
        n_layer=cfg_args["n_layer"],
        vocab_size=cfg_args["vocab_size"],
        ssm_cfg=ssm_cfg,
        d_intermediate=cfg_args.get("d_intermediate", 0),
        tie_embeddings=True,
    )
    model = MambaLMHeadModel(config)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: step={ckpt.get('step', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f}")
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--prompt", default="", help="Starting text")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint)
    model = model.to(device).eval()

    # Load tokenizer
    ckpt_dir = Path(args.checkpoint).parent
    tokenizer_path = ckpt_dir.parent / "data" / "tokenizer.json"
    # Try variations
    for p in [
        tokenizer_path,
        ckpt_dir / "tokenizer.json",
        Path(args.checkpoint).parent / f"{Path(args.checkpoint).stem}_tokenizer.json",
    ]:
        if p.exists():
            import json as _json
            data = _json.load(open(p, encoding="utf-8"))
            chars = data["chars"]
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}
            break
    else:
        # Fallback: try to find any tokenizer json
        data_dir = Path(args.checkpoint).parent.parent / "data"
        matches = list(data_dir.glob("*tokenizer*.json"))
        if matches:
            data = json.load(open(matches[0], encoding="utf-8"))
            chars = data["chars"]
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}
        else:
            print("Warning: no tokenizer found, using char-level from model vocab")
            # Minimal fallback
            chars = [chr(i) for i in range(256)]
            stoi = {c: i for i, c in enumerate(chars)}
            itos = {i: c for i, c in enumerate(chars)}  # type: ignore

    def encode(s):
        return [stoi.get(c, 0) for c in s]

    def decode(ids):
        return ''.join([itos.get(i, '') for i in ids])

    # Encode prompt
    if args.prompt:
        input_ids = torch.tensor([encode(args.prompt)], device=device)
    else:
        input_ids = torch.tensor([[random.randint(0, len(chars) - 1)]], device=device)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k})...\n")

    generated = input_ids
    with torch.no_grad():
        # Process prompt
        for _ in range(args.max_tokens):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(generated)
                next_logits = logits[:, -1, :] / args.temperature

                # Top-k sampling
                if args.top_k > 0:
                    vals, _ = next_logits.topk(args.top_k)
                    next_logits[next_logits < vals[:, -1:]] = float("-inf")

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)

    text = decode(generated[0].cpu().tolist())
    print(text)


if __name__ == "__main__":
    main()
