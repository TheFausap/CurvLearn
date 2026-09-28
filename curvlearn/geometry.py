"""Tensorised kappa-stereographic geometry in PyTorch.

A single real curvature ``k`` (a scalar tensor, possibly an ``nn.Parameter``) parameterises
the whole family: k<0 hyperbolic, k=0 Euclidean, k>0 spherical. Every op is smooth through
k=0 and is written to survive autograd differentiation w.r.t. ``k`` -- that gradient is the
object Design A studies.

Numerical policy (matches the memo's guidance):
  * all geometric ops run in float32 even under autocast (callers should wrap in
    ``torch.autocast(enabled=False)`` if training in bf16/f16);
  * arguments of atanh/tan are clamped away from their singularities;
  * points are projected into the model's domain before use.

The formulas mirror ``geometry_ref.py`` line for line; ``tests/test_geometry.py`` asserts
they agree with that dependency-free reference.
"""
from __future__ import annotations
import torch

EPS = 1e-9
TANH_CLAMP = 1.0 - 1e-6         # keep atanh argument < 1
MAX_NORM_FRAC = 1.0 - 1e-3      # points kept strictly inside the k<0 ball


def _sqrt_abs(k):
    return torch.sqrt(torch.clamp(k.abs(), min=EPS))


def artan_k(x, k):
    """Curvature-signed inverse tangent: atanh for k<0, atan for k>0, identity at k=0."""
    s = _sqrt_abs(k)
    neg = (k < 0)
    pos = (k > 0)
    out = x.clone()
    if neg.any():
        out = torch.where(neg, torch.atanh(torch.clamp(s * x, -TANH_CLAMP, TANH_CLAMP)) / s, out)
    if pos.any():
        out = torch.where(pos, torch.atan(s * x) / s, out)
    return out


def project(x, k):
    """Project points into the kappa-stereographic domain (only binds for k<0)."""
    n2 = x.pow(2).sum(-1, keepdim=True).clamp_min(EPS)
    max_n2 = torch.where(k < 0, (MAX_NORM_FRAC ** 2) / k.abs().clamp_min(EPS),
                         torch.full_like(k, float("inf")))
    scale = torch.where(n2 > max_n2, torch.sqrt(max_n2 / n2), torch.ones_like(n2))
    return x * scale


def mobius_add(x, y, k):
    """Moebius addition, broadcasting over leading dims; vectors on the last dim."""
    xy = (x * y).sum(-1, keepdim=True)
    xx = x.pow(2).sum(-1, keepdim=True)
    yy = y.pow(2).sum(-1, keepdim=True)
    num = (1 - 2 * k * xy - k * yy) * x + (1 + k * xx) * y
    den = (1 - 2 * k * xy + (k ** 2) * xx * yy).clamp_min(EPS)
    return num / den


def dist(x, y, k):
    """Geodesic distance d_k(x, y); returns 2||x-y|| at k=0. ``k`` is a scalar tensor."""
    kk = k.reshape(())                       # curvature is a single scalar throughout
    v = mobius_add(-x, y, kk)
    vnorm = v.pow(2).sum(-1).clamp_min(0).sqrt()
    s = torch.sqrt(torch.clamp(kk.abs(), min=EPS))
    d = (2.0 / s) * artan_k(vnorm, kk)
    return torch.where(kk.abs() < 1e-7, 2.0 * vnorm, d)


def expmap0(v, k):
    """Exponential map at the origin: tangent vector v -> manifold point."""
    vnorm = v.pow(2).sum(-1, keepdim=True).clamp_min(EPS).sqrt()
    s = _sqrt_abs(k)
    neg = (k < 0)
    pos = (k > 0)
    coef = torch.ones_like(vnorm)
    if neg.any():
        coef = torch.where(neg, torch.tanh(torch.clamp(s * vnorm, max=15.0)) / (s * vnorm), coef)
    if pos.any():
        coef = torch.where(pos, torch.tan(s * vnorm) / (s * vnorm), coef)
    return project(coef * v, k)


def logmap0(x, k):
    """Inverse of expmap0: manifold point -> tangent vector at the origin."""
    xnorm = x.pow(2).sum(-1, keepdim=True).clamp_min(EPS).sqrt()
    coef = artan_k(xnorm, k) / xnorm
    return coef * x


def pairwise_dist2(x, k):
    """Squared geodesic distances between all pairs along the sequence axis.

    x: (..., T, D)  ->  (..., T, T).  Used for distance-based attention scores.
    """
    xi = x.unsqueeze(-2)              # (..., T, 1, D)
    xj = x.unsqueeze(-3)              # (..., 1, T, D)
    d = dist(xi, xj, k)              # (..., T, T)
    return d * d
