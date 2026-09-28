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
    # Stay in the interior regime where neither implementation's boundary clamp binds
    # (s*||v|| stays < ~0.7 here, well under the 1-1e-3 atanh clamp), so torch and the
    # float64 reference agree to machine precision rather than differing at the boundary.
    random.seed(0)
    for _ in range(200):
        d = random.choice([2, 4, 8])
        k = random.uniform(-1.5, 1.5)
        x, y = _rand_vec(d, 0.15), _rand_vec(d, 0.15)
        ref = R.dist(x, y, k)
        got = float(G.dist(torch.tensor([x], dtype=torch.float64),
                           torch.tensor([y], dtype=torch.float64),
                           torch.tensor(k, dtype=torch.float64)))
        assert abs(ref - got) < 1e-6, (k, ref, got)


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


def test_self_distance_gradient_finite():
    # The attention diagonal is a point's distance to itself (== 0). sqrt(0) has an infinite
    # derivative, which used to NaN the first backward pass. Grad must now be finite.
    torch.manual_seed(0)
    x = (torch.randn(2, 5, 4) * 0.1).requires_grad_(True)   # (B, T, D)
    for kval in (-1.0, 0.0, 0.5):
        if x.grad is not None:
            x.grad = None
        d2 = G.pairwise_dist2(x, torch.tensor(kval))         # (B, T, T), diagonal included
        d2.sum().backward()
        assert torch.isfinite(x.grad).all(), (kval, x.grad)


def test_midpoint_flat_limit():
    # gyromidpoint at k->0 is the Euclidean weighted mean
    torch.manual_seed(0)
    pts = torch.randn(1, 5, 3) * 0.2               # (B, T, D)
    w = torch.rand(1, 2, 5)                          # (B, Q, T)
    w = w / w.sum(-1, keepdim=True)
    m = G.weighted_midpoint(pts, w, torch.tensor(1e-9))
    eucl = torch.matmul(w, pts)
    assert torch.allclose(m, eucl, atol=1e-4), (m, eucl)


def test_midpoint_idempotent():
    # all mass on copies of one point -> that point, at any curvature
    for kval in (-2.0, -1.0, -0.3, 0.0, 0.5):
        p = torch.tensor([[[0.2, -0.1, 0.05]]]).repeat(1, 4, 1)  # (1,4,3) identical
        w = torch.rand(1, 1, 4); w = w / w.sum(-1, keepdim=True)
        m = G.weighted_midpoint(p, w, torch.tensor(kval))
        assert torch.allclose(m[0, 0], p[0, 0], atol=1e-4), (kval, m)


def test_midpoint_matches_reference():
    import curvlearn.geometry_ref as R
    pts = [[0.1, 0.2], [-0.3, 0.05], [0.15, -0.1]]
    w = [0.5, 0.3, 0.2]
    for kval in (-2.0, -1.0, -0.3, 0.5):
        ref = R.weighted_midpoint(pts, w, kval)
        got = G.weighted_midpoint(torch.tensor([pts]), torch.tensor([[w]]), torch.tensor(kval))
        assert torch.allclose(got[0, 0], torch.tensor(ref), atol=1e-4), (kval, got, ref)


def test_expmap_gradient_finite_all_curvatures():
    # exp_0 runs project(); a torch.where(...,inf) there used to NaN the gradient for k>=0.
    # Check finite grads across hyperbolic, flat, and spherical curvatures.
    for kval in (-2.0, -1.0, -0.1, 0.0, 0.1, 0.5, 2.0):
        x = (torch.randn(3, 4) * 0.3).requires_grad_(True)
        G.expmap0(x, torch.tensor(kval)).pow(2).sum().backward()
        assert torch.isfinite(x.grad).all(), (kval, x.grad)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
