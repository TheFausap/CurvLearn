"""Pure-stdlib reference implementation of the kappa-stereographic model.

No numpy, no torch. This exists so the tensorised geometry in ``geometry.py`` can be
checked against a dependency-free ground truth (see ``tests/test_geometry.py``), and so a
mathematically-minded reader can inspect the formulas in isolation.

The kappa-stereographic model (Bachmann, Becigneul, Ganea 2020) unifies the hyperbolic
(kappa<0), Euclidean (kappa=0) and spherical (kappa>0) geometries in a single family that
is *analytic in kappa through zero* -- which is exactly what makes a learnable / scheduled
curvature well defined.

Verified facts (reproduced by the tests):
  * d_kappa(x, y) -> 2||x - y||           as kappa -> 0
  * d_kappa(0, y)^2 = 4 r^2 + (8/3)|kappa| r^4 + O(kappa^2),  r = ||y||
    so the curvature-gradient at the origin scales as ||x - y||^4;
  * away from the origin a conformal term ~ ||x||^2 ||x - y||^2 also appears, so the
    gradient vanishes at least as ||.||^2 with embedding scale. Either way it vanishes,
    which is the "flat is an attractor" mechanism.
"""
from __future__ import annotations
import math

EPS = 1e-9


def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _n2(a): return _dot(a, a)
def _scal(c, a): return [c * x for x in a]
def _add(a, b): return [x + y for x, y in zip(a, b)]


def mobius_add(x, y, k):
    """Moebius addition x (+)_k y in curvature k (any real)."""
    xy, xx, yy = _dot(x, y), _n2(x), _n2(y)
    num = _add(_scal(1 - 2 * k * xy - k * yy, x), _scal(1 + k * xx, y))
    den = 1 - 2 * k * xy + (k * k) * xx * yy
    return _scal(1.0 / den, num)


def dist(x, y, k):
    """Geodesic distance in the kappa-stereographic model, smooth through k = 0."""
    v = mobius_add(_scal(-1.0, x), y, k)
    r = math.sqrt(max(_n2(v), 0.0))
    if abs(k) < EPS:
        return 2.0 * r
    s = math.sqrt(abs(k))
    arg = s * r
    if k < 0:
        return (2.0 / s) * math.atanh(min(arg, 1 - EPS))
    return (2.0 / s) * math.atan(arg)


def dist2(x, y, k):
    d = dist(x, y, k)
    return d * d
