# CurvLearn — Results

A six-experiment study of tunable curvature in a small language model, on hierarchical
(Dyck-k) data. Each experiment is a falsifiable question; several forced a revision of the
hypothesis rather than confirming it. All geometry is verified against a dependency-free
reference (`tests/`), all sweeps are seeded, and every figure is reproducible from its notebook.

> **One-line finding.** Hyperbolic geometry helps sequence modelling in a *specific, characterised
> regime* — low embedding dimension, hierarchical data, and curvature applied to the
> **aggregation** (not the attention scores). The benefit is invisible at the dimensions people
> normally use, and in that regime the model reaches the useful curvature by ordinary training.

Figures referenced below live in `docs/figures/` — drop the PNGs the notebooks write to your
`OUTDIR` (Google Drive on Colab) into that folder to render them here.

---

## The curvature model

Representations live in the **κ-stereographic model**, one family that is analytic in the
curvature κ through zero (κ<0 hyperbolic, κ=0 Euclidean, κ>0 spherical). κ is an ordinary
learnable scalar (Euclidean AdamW), so we can study its *Euclidean* gradient directly.
Two attention mechanisms:

- **geodesic** — scores are `−β · d_κ(x_i, x_j)²` (curvature in the scores).
- **gyro** — tangent-space `Q·Kᵀ` scores, with curvature entering through a **hyperbolic
  gyromidpoint aggregation** of the value points. This matters because in the stereographic
  model `log₀ ∘ exp₀ = id`, so a tangent-space *weighted-sum* aggregation cancels κ out of the
  trunk entirely — curvature only does non-trivial work in the aggregation.

---

## A — Is flat an attractor? (`Design_A_flat_attractor.ipynb`)

Sweep initial curvature κ₀ × embedding scale; κ is a free learnable parameter.

![phase diagram](docs/figures/designA_phase_diagram.png)
![kappa trajectories](docs/figures/designA_kappa_trajectories.png)

- κ **freezes near its init at small embedding scale** and **escapes to ≈−0.67 at emb=1** (from
  every init, including κ₀=0). This is exactly the curvature-gradient scaling `∂L/∂κ ∝ ‖x‖⁴`
  near the origin (verified analytically and in `tests/`): small embeddings ⇒ vanishing κ
  gradient ⇒ frozen κ.
- **But the loss is flat in κ** (whole grid within 1.26–1.36; see `designA_loss.png`). So κ
  drifting to −0.67 is motion along a nearly-flat ridge, not the discovery of a better geometry.

**Verdict:** the flat-attractor *mechanism* is real, but on easy Dyck there is no better geometry
to reach — curvature is loss-neutral.

---

## A′ — Does deeper hierarchy want hyperbolic? (`Difficulty_sweep.ipynb`)

Fix κ, deepen the hierarchy, read **close-bracket loss** (the hierarchy-sensitive metric).

![difficulty sweep](docs/figures/difficulty_sweep.png)

Hyperbolic (κ<0) **hurts** at every difficulty and **worst on the hardest task** (relative-to-flat
+0.17…+0.19 for `hard`). The optimum is flat-to-mildly-spherical. **The opposite of the naive
prediction.**

---

## A″ — Is the geometry itself sound? (`Tree_embedding.ipynb`)

Drop the LM; embed tree nodes as κ-stereographic points and fit geodesic distance to tree
distance (the task hyperbolic geometry is built for). Metric: average multiplicative distortion.

![tree embedding](docs/figures/tree_embed.png)

Hyperbolic **halves** the distortion vs Euclidean (~0.08 → ~0.03), and the advantage **grows with
depth** (hyperbolic/Euclidean distortion ratio 0.43 / 0.36 / 0.27 for shallow/medium/deep). So
the κ-stereographic implementation is correct and delivers the textbook advantage.

**Contrast A′ vs A″:** the geometry embeds hierarchy well, but folding κ into distance-based
attention scores destroys the benefit. **The open problem is the mechanism, not the space.**

---

## A‴ — The mechanism fix: place, don't score (`Mechanism_compare.ipynb`)

Compare `geodesic` vs `gyro` on the difficulty sweep.

![mechanism comparison](docs/figures/mechanism_compare.png)

`gyro` doesn't yet beat flat at d_model=32, but it **halves the hard-task hyperbolic penalty**
(+0.17…+0.19 → +0.06…+0.10) and **flips the depth ordering**: under `gyro`, `hard` has the
*smallest* penalty (it was the largest under `geodesic`). The mechanism helps most exactly where
the hierarchy is deepest — the right fingerprint.

---

## A⁗ — Dimension is the knob (`Dimension_sweep.ipynb`)

Sweep d_model ∈ {4, 8, 16, 32} for `gyro`; "too much space goes flat".

![dimension sweep](docs/figures/dim_sweep.png)

**The crossover.** Hyperbolic aggregation **significantly beats flat at small dimension** —
medium at d=4, hard at d=8 (seed error bands fully below zero) — and **reverses at d=16–32**. The
"advantage dimension" scales with hierarchy complexity (the bigger tree needs a few more dims
before curvature pays). This is the regime where tunable curvature genuinely helps a sequence
model, and it is masked at the dimensions normally used.

---

## B — Can the model learn/schedule the useful curvature? (`Design_B.ipynb`)

Five settings in the crossover regime (gyro, medium@d=4, hard@d=8): `flat`, `fixed_opt` (κ=−0.5),
`learn_from_opt` (init −1), `learn_from_flat` (init 0), `anneal` (κ: 0→−1 schedule).

![design B](docs/figures/design_b.png)

- **Every curvature setting beats flat** in both regimes (~0.05–0.07 lower close-loss).
- **The model learns the useful curvature unaided from a flat start**: both `learn_from_opt` and
  `learn_from_flat` converge to κ≈−0.45 (near the −0.5 optimum) regardless of init. The flat
  attractor does **not** trap it here.
- The schedule is competitive but **not necessary** (regime-dependent which of schedule / free
  learning wins).

---

## The unifying result

A single axis — **curvature-gradient strength** — governs both halves of the story:

| regime | κ dynamics | does curvature help? |
|---|---|---|
| weak gradient (small embeddings, Design A) | κ frozen at init | no — loss-neutral |
| strong gradient (gyro, low dimension) | κ escapes flat, finds the optimum | yes — beats flat |

Flat is an attractor **only** where curvature is also loss-neutral; where curvature helps, the
gradient is strong enough that the model finds it on its own. That reconciles A (κ freezes /
loss-neutral) with B (κ learned freely / curvature helps).

**Caveats.** Gains are modest (~0.05–0.07 close-bracket-loss units) on synthetic Dyck at small
scale; `kappa_lr=5e-2` overshoots early then recovers (a lower LR / warmup would clean this up);
"close-bracket loss" is the sensitive metric, not overall loss. This is a proof of principle in a
characterised toy regime, not a scaling claim.

## Reproduce

```bash
pip install -e .
python tests/test_geometry.py && python tests/test_trees.py
# then run any notebook in notebooks/, or e.g.
python -m curvlearn.dim_sweep --out results/dim_sweep_results.json
python -m curvlearn.plot  # writes the Design-A figures
```

## Next

- Tune `kappa_lr` / add warmup to remove the κ overshoot.
- Scale with proportionally small *per-head* dimension (the crossover is about per-head packing).
- A real hierarchical corpus (code / language) instead of Dyck.
- **Per-subspace mixed curvature** — a product manifold 𝕊×𝔼×ℍ with a learnable curvature per
  head, so the model can choose different geometry in different subspaces.
