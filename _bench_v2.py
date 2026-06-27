import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, '.')
device = torch.device("cuda")
from mamba3_ssm import MambaLMHeadModel, MambaConfig, CONFIGS

free, total = torch.cuda.mem_get_info()
print(f"VRAM: {total/1e9:.1f}GB free={free/1e9:.1f}GB", flush=True)

for name in ["small", "medium", "large"]:
    cfg = CONFIGS[name]
    mc = MambaConfig(
        d_model=cfg["d_model"], n_layer=cfg["n_layer"], vocab_size=256,
        ssm_cfg={"d_state": cfg["d_state"], "expand": 2, "headdim": 64,
                 "is_mimo": True, "mimo_rank": 4},
        tie_embeddings=True,
    )
    d = cfg["desc"]
    print(f"\n--- {name} ({d}) ---", flush=True)
    try:
        model = MambaLMHeadModel(mc).to(device)
    except RuntimeError as e:
        print(f"  OOM on creation: {e}", flush=True)
        continue
    p = sum(p.numel() for p in model.parameters())
    print(f"  Params: {p/1e6:.0f}M", flush=True)

    # Warmup: 3 forward-only + 3 forward+backward
    for i in range(3):
        x = torch.randint(0, 256, (cfg["batch_size"], cfg["seq_len"]), device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _ = model(x)
        print(f"  warmup fwd {i}", flush=True)

    model.zero_grad()
    for i in range(3):
        x = torch.randint(0, 256, (cfg["batch_size"], cfg["seq_len"]), device=device)
        y = torch.randint(0, 256, (cfg["batch_size"], cfg["seq_len"]), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        loss.backward()
        model.zero_grad()
        print(f"  warmup step {i}", flush=True)

    # Timed trials
    times = []
    for t in range(5):
        xi = torch.randint(0, 256, (cfg["batch_size"], cfg["seq_len"]), device=device)
        yi = torch.randint(0, 256, (cfg["batch_size"], cfg["seq_len"]), device=device)
        model.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xi)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), yi.view(-1))
        loss.backward()
        torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t0
        times.append(t_elapsed)
        print(f"  trial {t}: {t_elapsed*1000:.0f}ms loss={loss.item():.4f}", flush=True)

    avg = sum(times) / len(times)
    tok_s = cfg["batch_size"] * cfg["seq_len"] / avg
    step_ms = avg * cfg["grad_accum"] * 1000

    TOK = 50_000_000
    train_tok = int(0.95 * TOK)
    eff_bs = cfg["batch_size"] * cfg["grad_accum"]
    tps_ = cfg["seq_len"] * eff_bs
    spe = max(1, train_tok // tps_)
    total_s = spe * 3 * step_ms / 1000
    h, rem = divmod(total_s, 3600)
    m, _ = divmod(rem, 60)
    print(f"  AVG: {avg*1000:.0f}ms/micro | {tok_s:.0f} tok/s | Step: {step_ms:.0f}ms")
    print(f"  TinyStories 3ep: ~{int(h)}h {int(m)}m ({spe:,} steps/ep)", flush=True)

    del model
    torch.cuda.empty_cache()
