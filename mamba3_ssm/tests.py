"""
Sanity checks and tests for the Mamba-3 implementation.

Run with: python -m pytest tests/ -v
Or simply: python -m mamba3_ssm.tests.run_tests
"""

import torch
from mamba3_ssm.layer import Mamba3
from mamba3_ssm.block import MambaLMHeadModel
from mamba3_ssm.config import MambaConfig
from mamba3_ssm.utils import count_parameters


def test_siso_forward_shape():
    """SISO forward should preserve (B, L, d_model) shape."""
    model = Mamba3(d_model=256, d_state=64, expand=2, headdim=32, ngroups=1, is_mimo=False)
    x = torch.randn(2, 128, 256)
    y = model(x)
    assert y.shape == x.shape, f"SISO forward shape mismatch: {y.shape} != {x.shape}"
    print("  PASS: SISO forward shape")


def test_mimo_forward_shape():
    """MIMO forward should preserve (B, L, d_model) shape."""
    model = Mamba3(d_model=256, d_state=64, expand=2, headdim=32, ngroups=1,
                   is_mimo=True, mimo_rank=4)
    x = torch.randn(2, 128, 256)
    y = model(x)
    assert y.shape == x.shape, f"MIMO forward shape mismatch: {y.shape} != {x.shape}"
    print("  PASS: MIMO forward shape")


def test_siso_step_shape():
    """SISO .step should produce (B, d_model) output and correct state shapes."""
    model = Mamba3(d_model=256, d_state=64, expand=2, headdim=32, ngroups=1, is_mimo=False)
    angle_s, h_s, bx_s = model.allocate_inference_cache(batch_size=2)
    u = torch.randn(2, 256)

    # Check state shapes
    assert angle_s.shape == (2, model.nheads, model.num_rope_angles)
    assert h_s.shape == (2, model.nheads, model.headdim, model.d_state)
    assert bx_s.shape == (2, model.nheads, model.headdim, model.d_state)

    o, angle_s, h_s, bx_s = model.step(u, angle_s, h_s, bx_s)
    assert o.shape == (2, 256), f"SISO step output shape mismatch: {o.shape}"
    print("  PASS: SISO step shapes")


def test_mimo_step_shape():
    """MIMO .step should produce (B, d_model) output; state has no P dimension."""
    model = Mamba3(d_model=256, d_state=64, expand=2, headdim=32, ngroups=1,
                   is_mimo=True, mimo_rank=4)
    angle_s, h_s, bx_s = model.allocate_inference_cache(batch_size=2)
    u = torch.randn(2, 256)

    # MIMO state: (B, H, D) — no headdim dimension
    assert h_s.shape == (2, model.nheads, model.d_state), \
        f"MIMO state shape: {h_s.shape}"
    assert bx_s.shape == (2, model.nheads, model.d_state)

    o, angle_s, h_s, bx_s = model.step(u, angle_s, h_s, bx_s)
    assert o.shape == (2, 256)
    print("  PASS: MIMO step shapes")


def test_deterministic_step():
    """Step-by-step decode should match forward pass (same sequence)."""
    model = Mamba3(d_model=128, d_state=32, expand=2, headdim=32, is_mimo=False)
    model.eval()

    x = torch.randn(1, 8, 128)
    with torch.no_grad():
        y_forward = model(x)

    # Now feed tokens one by one via .step
    angle_s, h_s, bx_s = model.allocate_inference_cache(batch_size=1)
    with torch.no_grad():
        for t in range(x.shape[1]):
            u = x[:, t]   # (1, d_model)
            out, angle_s, h_s, bx_s = model.step(u, angle_s, h_s, bx_s)
            # Compare with forward output at position t
            diff = (out - y_forward[:, t]).abs().max().item()
            assert diff < 1e-4, \
                f"Step {t} mismatch: max diff = {diff:.6f}"

    print("  PASS: deterministic step-by-step matches forward")


def test_mimo_deterministic_step():
    """MIMO: step-by-step decode should match forward pass."""
    model = Mamba3(d_model=128, d_state=32, expand=2, headdim=32,
                   is_mimo=True, mimo_rank=2)
    model.eval()

    x = torch.randn(1, 8, 128)
    with torch.no_grad():
        y_forward = model(x)

    angle_s, h_s, bx_s = model.allocate_inference_cache(batch_size=1)
    with torch.no_grad():
        for t in range(x.shape[1]):
            out, angle_s, h_s, bx_s = model.step(x[:, t], angle_s, h_s, bx_s)
            diff = (out - y_forward[:, t]).abs().max().item()
            assert diff < 1e-4, \
                f"MIMO step {t} mismatch: max diff = {diff:.6f}"

    print("  PASS: MIMO deterministic step-by-step matches forward")


def test_full_lm_model():
    """Full MambaLMHeadModel forward should produce correct logit shape."""
    cfg = MambaConfig(
        d_model=256,
        n_layer=4,
        vocab_size=1000,
        ssm_cfg={"d_state": 64, "expand": 2, "headdim": 32, "is_mimo": False},
    )
    model = MambaLMHeadModel(cfg)
    input_ids = torch.randint(0, 1000, (2, 64))
    logits = model(input_ids)
    assert logits.shape == (2, 64, cfg.vocab_size + (8 - cfg.vocab_size % 8) % 8)
    trainable, total = count_parameters(model)
    assert trainable == total  # all params should be trainable
    print(f"  PASS: Full LM model ({total:,} params)")


def test_parameter_counting():
    """Verify parameter counts for both SISO and MIMO models."""
    model_siso = Mamba3(d_model=128, d_state=32, expand=2, headdim=32, is_mimo=False)
    model_mimo = Mamba3(d_model=128, d_state=32, expand=2, headdim=32,
                        is_mimo=True, mimo_rank=4)
    t_s, _ = count_parameters(model_siso)
    t_m, _ = count_parameters(model_mimo)
    # MIMO has extra mimo_x and mimo_o parameters
    assert t_m > t_s, f"MIMO ({t_m}) should have more params than SISO ({t_s})"
    print(f"  PASS: SISO={t_s:,}  MIMO={t_m:,}")


def test_rope_fraction_1():
    """rope_fraction=1.0 should work (all state dims rotate)."""
    model = Mamba3(d_model=128, d_state=64, expand=2, headdim=32, rope_fraction=1.0)
    assert model.num_rope_angles == 32  # 64 / 2
    x = torch.randn(1, 16, 128)
    y = model(x)
    assert y.shape == x.shape
    print("  PASS: rope_fraction=1.0 works")


def test_gradient_flow():
    """Parameters should receive gradients after backward pass."""
    model = Mamba3(d_model=128, d_state=32, expand=2, headdim=32, is_mimo=False)
    x = torch.randn(2, 16, 128)
    y = model(x)
    loss = y.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"
    print("  PASS: gradient flow OK")


if __name__ == "__main__":
    torch.manual_seed(42)
    print("Running Mamba-3 sanity checks...")
    print()

    tests = [
        test_siso_forward_shape,
        test_mimo_forward_shape,
        test_siso_step_shape,
        test_mimo_step_shape,
        test_deterministic_step,
        test_mimo_deterministic_step,
        test_full_lm_model,
        test_parameter_counting,
        test_rope_fraction_1,
        test_gradient_flow,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed == 0:
        print("All checks passed!")
