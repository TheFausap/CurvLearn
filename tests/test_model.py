"""Forward/backward smoke tests for the models. Run: python tests/test_model.py"""
import torch
from curvlearn.model import ModelConfig, GeometricAttentionLM, ProductGyroLM


def _batch(vocab=6, B=3, T=16):
    x = torch.randint(0, vocab, (B, T))
    y = torch.randint(0, vocab, (B, T))
    return x, y


def test_geometric_lm_modes_finite():
    for mode in ("geodesic", "gyro"):
        cfg = ModelConfig(vocab_size=6, d_model=8, n_heads=2, n_layers=2, attention_mode=mode)
        m = GeometricAttentionLM(cfg)
        x, y = _batch()
        _, loss = m(x, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(m.kappa.grad)


def test_product_lm_shared_and_per_head():
    for nf_name, nf in (("shared", 1), ("per_head", 4)):
        cfg = ModelConfig(vocab_size=6, d_model=8, n_heads=4, n_layers=2)
        m = ProductGyroLM(cfg, n_curv_factors=nf)
        assert m.kappa.numel() == nf
        x, y = _batch()
        _, loss = m(x, y)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(m.kappa.grad).all()
        assert m.kappa.grad.numel() == nf          # every per-head curvature gets a gradient


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
