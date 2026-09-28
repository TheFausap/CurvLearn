"""Cross-check the tensorised geometry against the dependency-free reference, and pin the
two facts Design A rests on. Run with `pytest -q` or `python tests/test_geometry.py`."""
import math
import random
import torch

from curvlearn import geometry as G
from curvlearn import geometry_ref as R


def _rand_vec(d, scale):
    return [random.uniform(-scale, scale) for _ in range(d)]


def test_matches_reference():
    random.seed(0)
    for _ in range(200):
        d = random.choice([2, 4, 8])
        k = random.uniform(-2.0, 2.0)
        x, y = _rand_vec(d, 0.2), _rand_vec(d, 0.2)
        ref = R.dist(x, y, k)
        got = float(G.dist(torch.tensor([x]), torch.tensor([y]), torch.tensor(k)))
        assert abs(ref - got) < 1e-4, (k, ref, got)


def test_flat_limit():
    x, y = torch.tensor([[0.1, -0.2, 0.05]]), torch.tensor([[0.15, 0.1, -0.1]])
    eu = 2 * (x - y).norm()
    d = G.dist(x, y, torch.tensor(1e-8))
    assert torch.allclose(d, eu, atol=1e-4)


def test_origin_gradient_coefficient():
    # d^2(0, y) = 4 r^2 + (8/3)|k| r^4 + O(k^2);  d(d^2)/d|k| at 0 == (8/3) r^4.
    # Finite-differencing squared distances: the signal ~ (8/3) h r^4 sits on top of 4 r^2,
    # so it underflows float32 (e.g. 1.7e-10 vs 0.01). Verify the identity in float64.
    dt = torch.float64
    h = 1e-5
    for r in [0.05, 0.1, 0.2, 0.4]:
        x0 = torch.zeros(1, 3, dtype=dt)
        y0 = torch.tensor([[r, 0.0, 0.0]], dtype=dt)
        d2_0 = G.dist(x0, y0, torch.tensor(0.0, dtype=dt)) ** 2
        d2_m = G.dist(x0, y0, torch.tensor(-h, dtype=dt)) ** 2
        # k = -h has |k| = h > 0, so d^2 grows: d(d^2)/d|k| = +(8/3) r^4
        g = float((d2_m - d2_0) / h)
        pred = (8.0 / 3.0) * r ** 4
        assert abs(g - pred) / pred < 0.05, (r, g, pred)


def test_curvature_gradient_vanishes_with_scale():
    # autograd d(loss)/d kappa shrinks as embeddings shrink -> flat is an attractor
    torch.manual_seed(0)
    grads = []
    for scale in [1.0, 0.3, 0.1]:
        x = (torch.randn(16, 3) * scale)
        y = (torch.randn(16, 3) * scale)
        k = torch.tensor(-0.5, requires_grad=True)
        loss = (G.dist(x, y, k) ** 2).mean()
        loss.backward()
        grads.append(abs(float(k.grad)))
    assert grads[0] > grads[1] > grads[2], grads


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
