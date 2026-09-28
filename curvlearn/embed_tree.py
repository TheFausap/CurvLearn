"""Tree-distance embedding test -- does hyperbolic curvature actually help, when used the way
the theory intends (embedding placement, not attention scores)?

Embed the nodes of a tree as kappa-stereographic points and fit geodesic distances to tree
distances. Metric = average multiplicative distortion after the optimal global scale (Sala et
al.): distortion = mean_{i<j} | d_geo(i,j) / (s* * d_tree(i,j)) - 1 |, lower is better.

Loss is the variance of the log-ratio log d_geo - log d_tree: minimizing it forces d_geo
proportional to d_tree with ANY global scale (the scale is a nuisance the metric factors out),
and -- unlike an absolute distance MSE -- it does not collapse to the degenerate all-points-
at-the-origin solution.

Prediction if the geometry works: for TREES, distortion at kappa<0 (hyperbolic) is lower than
at kappa=0 (Euclidean), and the gap GROWS with tree depth and SHRINKS with embedding dimension.
If hyperbolic wins here but lost the Dyck LM sweep, the LM negative result is about the
attention mechanism, not the geometry.
"""
from __future__ import annotations
import argparse, json, math, os, time
import yaml
import torch

from . import geometry as G
from .trees import balanced_tree


def train_embedding(D, kappa, dim, steps, lr, seed, device):
    torch.manual_seed(seed)
    N = D.shape[0]
    X = torch.nn.Parameter(torch.randn(N, dim, device=device) * 1e-3)   # tangent params
    k = torch.tensor(float(kappa), device=device)
    opt = torch.optim.Adam([X], lr=lr)
    iu = torch.triu_indices(N, N, offset=1, device=device)
    Dt = D.to(device)[iu[0], iu[1]]
    logD = torch.log(Dt)

    def geo_dists():
        Xp = G.expmap0(X, k)                                  # tangent -> manifold
        d2 = G.pairwise_dist2(Xp.unsqueeze(0), k)[0]          # (N, N)
        return torch.sqrt(d2[iu[0], iu[1]] + 1e-12)

    for _ in range(steps):
        r = torch.log(geo_dists() + 1e-9) - logD
        loss = r.var(unbiased=False)                          # scale-free, non-degenerate
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([X], 1.0)
        opt.step()

    with torch.no_grad():
        r = torch.log(geo_dists() + 1e-9) - logD
        c = r.mean()                                          # optimal log global scale
        distortion = (torch.exp(r - c) - 1).abs().mean().item()
        return {"distortion": distortion, "loss": float(r.var(unbiased=False))}


def _atomic_dump(obj, path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def default_spec():
    return {
        "trees": [
            {"name": "shallow", "b": 2, "h": 4},   # N=31
            {"name": "medium",  "b": 2, "h": 6},   # N=127
            {"name": "deep",    "b": 2, "h": 8},   # N=511
        ],
        "kappa_grid": [-3.0, -2.0, -1.0, -0.5, -0.2, 0.0, 0.5],
        "seeds": 3,
        "dim": 10,          # lower this (5, 2) to amplify the hyperbolic advantage
        "steps": 1500,
        "lr": 5e-2,
        "device": "cuda",
    }


def run_tree_sweep(spec, checkpoint_path=None, resume=True):
    dev = spec.get("device", "cuda")
    dev = dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu"
    dim, steps, lr = spec["dim"], spec["steps"], spec["lr"]
    trees, kappa_grid, n_seeds = spec["trees"], spec["kappa_grid"], spec["seeds"]

    records, done = [], set()
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            prev = json.load(open(checkpoint_path))
            records = [r for r in prev.get("records", [])
                       if math.isfinite(r.get("distortion", float("nan")))]
            done = {(r["tree"], round(r["kappa"], 6), r["seed"]) for r in records}
            print(f"resuming: {len(done)} runs cached", flush=True)
        except Exception as e:
            print(f"could not resume ({e}); fresh", flush=True)

    total = len(trees) * len(kappa_grid) * n_seeds
    t0, i = time.time(), 0
    for t in trees:
        N, D = balanced_tree(t["b"], t["h"])
        for kap in kappa_grid:
            for seed in range(n_seeds):
                i += 1
                key = (t["name"], round(kap, 6), seed)
                if key in done:
                    print(f"[{i}/{total}] {t['name']} k={kap:+.1f} s={seed} (cached)", flush=True)
                    continue
                res = train_embedding(D, kap, dim, steps, lr, seed, dev)
                rec = {"tree": t["name"], "b": t["b"], "h": t["h"], "N": N,
                       "kappa": kap, "seed": seed, "dim": dim,
                       "distortion": res["distortion"], "loss": res["loss"]}
                records.append(rec)
                if checkpoint_path:
                    _atomic_dump({"spec": spec, "records": records}, checkpoint_path)
                print(f"[{i}/{total}] {t['name']:7s}(N={N}) k={kap:+.1f} s={seed} "
                      f"distortion={res['distortion']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    return {"spec": spec, "records": records, "wall_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/tree_embed_results.json")
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    spec = default_spec()
    if args.config and os.path.exists(args.config):
        spec = yaml.safe_load(open(args.config))
    if args.dim is not None:
        spec["dim"] = args.dim
    if args.device is not None:
        spec["device"] = args.device
    out = run_tree_sweep(spec, checkpoint_path=args.out, resume=not args.no_resume)
    _atomic_dump(out, args.out)
    print(f"\nwrote {args.out} ({len(out['records'])} runs)")


if __name__ == "__main__":
    main()
