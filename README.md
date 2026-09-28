# CurvLearn

**Tunable-curvature representation spaces for language models** — where the geometry the
hidden vectors live in is not fixed to Euclidean, but starts non-flat and is *learned or
scheduled during training*.

This repository works through a four-step program (see `docs/` and the design memo). The
current release implements **Design A**, the seed experiment that the rest is built on.

> **The thesis in one line.** A freely-learnable curvature initialised at flat (κ = 0)
> barely moves — flat is an attractor — so a non-Euclidean geometry has to be *put there on
> purpose* and held off flat. Design A measures the attractor; Designs B–D exploit escaping it.

---

## Why flat is an attractor (the one piece of theory you need)

Work in the **κ-stereographic model**, the single family that is analytic in the curvature κ
through zero (κ<0 hyperbolic, κ=0 Euclidean, κ>0 spherical). Expanding the squared geodesic
distance about the origin,

```
d_κ(0, y)^2  =  4 r^2  +  (8/3) |κ| r^4  +  O(κ^2),      r = ||y||.
```

So the gradient of the loss with respect to the curvature, at κ = 0, scales as **r⁴** at the
origin — and off the origin an additional conformal term of order `||x||² ||x−y||²` appears,
so in the worst case it vanishes as **r²**. Either way:

- under LayerNorm / RMSNorm / unit-sphere embeddings, `r` is small, so `∂L/∂κ ≈ 0` at init —
  **κ receives almost no gradient and stays where it started**;
- for purely distance-based losses the leading term is even in κ, making κ = 0 a genuine
  **saddle** you cannot leave by gradient descent.

The prescription is therefore structural, not a hyperparameter afterthought: **initialise κ
away from 0**, **scale embeddings up**, and **give κ its own, larger learning rate**. All
three are defaults in this code. The claim is pinned by `tests/test_geometry.py`
(the `(8/3) r⁴` coefficient is asserted to 5%).

Full treatment — the three distinct meanings of "curvature" in this literature, the
occupancy map of what has and hasn't been tried, and the five ranked designs — is in the
companion design memo (`tunable_curvature_memo.md`, shared alongside this repo).

---

## The roadmap

| step | question | status |
|---|---|---|
| **A** | Does a free κ actually move, or is flat an attractor? Map the basin. | **implemented** |
| B | Curvature *annealing curriculum*: start non-flat, drive |κ| on a schedule on a product 𝕊×𝔼×ℍ manifold. | hook in place (`kappa_mode="schedule"`) |
| C | *Depth-scheduled* curvature: one κ per layer, hyperbolic→flat across depth. | planned |
| D | Pseudo-Riemannian (indefinite-signature) residual stream. | planned |

Design A is the falsifiable premise-check; only run B–D once A shows the basin is real.

---

## Install

```bash
git clone https://github.com/TheFausap/CurvLearn.git
cd CurvLearn
pip install -r requirements.txt          # torch, numpy, matplotlib, pyyaml
```

A100 / Colab: open `notebooks/Design_A_flat_attractor.ipynb` and run top to bottom.

## Run Design A

```bash
bash scripts/run_designA.sh                       # sweep + plot, uses experiments/designA.yaml
# or step by step:
python -m curvlearn.sweep --config experiments/designA.yaml --out results/designA_results.json
python -m curvlearn.plot  --results results/designA_results.json --outdir results
```

Outputs:

- `results/designA_phase_diagram.png` — a heatmap of `|κ_final − κ₀|` over the
  (initial curvature × embedding scale) grid. **Predicted:** a dark (frozen) basin around
  κ₀ = 0 that narrows as the embedding scale grows.
- `results/designA_kappa_trajectories.png` — κ over training for every cell; the flat-init
  runs should sit on their starting line while the strongly-curved ones migrate.

The default sweep is 27 cells × 2000 steps of a 2-layer, d=64 model on Dyck-3 — minutes on an
A100, and it will complete on CPU. Swap in a real corpus with `dataset: {name: char, path: …}`.

**Persistence & resume.** The sweep writes its results file *after every cell*, atomically, and
skips cells already present on a re-run — so a killed process (or a recycled Colab runtime)
costs at most one cell. Point `--out` at a durable location to resume across restarts:

```bash
python -m curvlearn.sweep --out /content/drive/MyDrive/CurvLearn/results/designA_results.json
# re-run the same command after a disconnect -> it continues; add --no-resume to start over
```

In `notebooks/Design_A_flat_attractor.ipynb` this is wired to Google Drive automatically: the
notebook mounts Drive and sets `OUTDIR=/content/drive/MyDrive/CurvLearn/results`, so results and
figures survive the runtime being recycled.

## What to look for

1. **Does κ move from κ₀ = 0?** If it does not (small `|Δκ|` in the middle columns) while it
   *does* move for |κ₀| ≥ some threshold, the attractor is confirmed.
2. **Does the basin shrink with embedding scale?** The r²–r⁴ argument predicts the frozen
   region narrows as `emb_scale` grows. That two-way dependence is the real signature.
3. **Where does escaped κ settle, and does lower loss track the data's true geometry?**
   Dyck is hierarchical, so escaped runs should prefer κ < 0 (hyperbolic).

## Repository layout

```
curvlearn/
  geometry.py       kappa-stereographic ops in torch (mobius add, dist, exp/log maps), fp32 + clamps
  geometry_ref.py   pure-stdlib reference of the same formulas (no deps) -- ground truth for tests
  model.py          GeometricAttentionLM: geodesic-distance attention + geodesic output head
  data.py           Dyck-k hierarchy generator (default) + char-level loader
  train.py          instrumented run: logs kappa, its gradient, and loss; 3 curvature modes
  sweep.py          Design A: the kappa0 x emb_scale sweep
  plot.py           phase diagram + trajectory figures
experiments/designA.yaml     the sweep spec
tests/test_geometry.py       torch-vs-reference agreement; the (8/3)r^4 law; gradient vanishing
notebooks/Design_A_flat_attractor.ipynb   Colab / A100 runner
```

## Numerical notes (they matter here)

- All geometric ops run in **float32** even under autocast; the model wraps them in an
  `autocast(enabled=False)` block. atanh/tan arguments are clamped; points are projected into
  the model domain.
- The **hyperboloid/Lorentz** model is more stable than the Poincaré ball at strong negative
  curvature; this release uses the κ-stereographic (ball-type) form for a clean κ→0 limit —
  keep |κ| ≤ ~2 and watch for NaNs on the spherical (κ>0) side, where `tan` can blow up.
- κ is an ordinary Euclidean parameter on AdamW, **not** a Riemannian optimizer — Design A is
  about its Euclidean gradient, and a Riemannian update would change the dynamics under study.

## License

MIT — see `LICENSE`.
