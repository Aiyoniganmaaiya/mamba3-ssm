"""
generate.py — Text generation with a trained Mamba-3 checkpoint.

Uses incremental decode (O(L) per step) via model.step().

Usage:
  python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"
  python generate.py --checkpoint checkpoints/best.pt --prompt "Hello" --max-tokens 500 --temperature 0.7
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from mamba3_ssm import MambaLMHeadModel, MambaConfig
from mamba3_ssm.tokenizer import load_tokenizer


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_args = ckpt["config"]

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
    model = model.to(device).eval()
    val_loss = ckpt.get("val_loss")
    val_str = f"{val_loss:.4f}" if val_loss is not None else "?"
    print(f"Loaded: step={ckpt.get('step', '?')}, val_loss={val_str}")
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)

    # Load tokenizer from data dir
    data_dir = Path(args.checkpoint).parent.parent / "data"
    tokenizer, tok_type = load_tokenizer(data_dir, config.vocab_size)
    print(f"Tokenizer: {tok_type} ({tokenizer.vocab_size} tokens)")

    # Encode prompt
    if args.prompt:
        input_ids = tokenizer.encode(args.prompt)
    else:
        input_ids = [random.randint(0, tokenizer.vocab_size - 1)]
    input_tensor = torch.tensor([input_ids], device=device)

    # Allocate cache and process prompt token-by-token
    cache = model.allocate_inference_cache(1)
    for i in range(input_tensor.shape[1]):
        logits, cache = model.step(input_tensor[:, i], cache)

    # Generate new tokens
    generated = input_ids[:]
    for _ in range(args.max_tokens):
        logits = logits / args.temperature
        if args.top_k > 0:
            vals, _ = logits.topk(args.top_k)
            logits[logits < vals[:, -1:]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        generated.append(next_token.item())
        logits, cache = model.step(next_token, cache)

    text = tokenizer.decode(generated)
    print(f"\nPrompt: {args.prompt!r}")
    print(f"---\n{text}\n---")


if __name__ == "__main__":
    main()
