"""Difficulty sweep: is curvature ever loss-relevant?

Design A showed kappa moves (or freezes) but is loss-neutral on easy Dyck-3. Before scheduling
kappa (Design B) is worth anything, we need a task whose loss actually DEPENDS on kappa. This
sweep FIXES kappa at a grid of values (kappa_mode='fixed' -- no learnable-kappa dynamics to
confound the reading) and measures the loss at increasing hierarchical difficulty, over seeds.

Two design choices that make the test sensitive:
  * metric = loss on CLOSE-bracket positions only. Open brackets are near-uniform and easy;
    the hierarchy bites exactly when a close bracket must match an opener that can be
    arbitrarily far back (sequence distance != tree distance). Overall loss washes this out.
  * small d_model. Hyperbolic space's low-distortion embedding of trees (exponential node
    growth ~ exponential volume) helps most when the Euclidean dimension is a binding
    constraint. A large d_model can pack the stack states flat and hide any curvature effect.

Read the output as: does a hyperbolic dip (close-loss lower at kappa<0) EMERGE as difficulty
grows, while easy stays flat in kappa? If yes, curvature is loss-relevant and Design B has a
target. If it stays flat at every difficulty, tunable curvature does not help this family.
"""
from __future__ import annotations
import argparse, json, math, os, time
import yaml
import torch
import torch.nn.functional as F

from .model import ModelConfig
from .train import TrainConfig, run_training
from .data import make_dataset


def eval_losses(model, data_eval, seq_len, batches=6, batch_size=64):
    """Overall and close-bracket next-token loss on a fixed eval stream."""
    dev = next(model.parameters()).device
    k = getattr(data_eval, "k", None)
    model.eval()
    tot, close = [], []
    with torch.no_grad():
        for _ in range(batches):
            x, y = data_eval.batch(batch_size, seq_len, device=dev)
            logits, _ = model(x, y)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1), reduction="none")
            tot.append(ce.mean().item())
            if k is not None:
                m = (y.reshape(-1) >= k)               # close-bracket targets have id >= k
                if m.any():
                    close.append(ce[m].mean().item())
    model.train()
    return (sum(tot) / len(tot),
            (sum(close) / len(close) if close else float("nan")))


def _atomic_dump(obj, path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def default_spec():
    return {
        "difficulties": [
            {"name": "easy",   "k": 3, "max_depth": 8},
            {"name": "medium", "k": 3, "max_depth": 25},
            {"name": "hard",   "k": 5, "max_depth": 50},
        ],
        "kappa_grid": [-2.0, -1.0, -0.5, 0.0, 0.5],
        "seeds": 3,
        "emb_scale": 1.0,
        "attention_modes": ["geodesic", "gyro"],
        "model": {"d_model": 32, "n_layers": 2, "n_heads": 1, "geodesic_output": True},
        "train": {"steps": 2000, "batch_size": 64, "seq_len": 128,
                  "lr": 3e-3, "device": "cuda", "log_every": 50},
    }


def run_difficulty(spec, checkpoint_path=None, resume=True):
    diffs = spec["difficulties"]
    kappa_grid = spec["kappa_grid"]
    n_seeds = spec["seeds"]
    emb = spec.get("emb_scale", 1.0)
    modes = spec.get("attention_modes", ["geodesic"])
    base_model = {k: v for k, v in spec.get("model", {}).items()
                  if k not in ("seq_len", "attention_mode")}
    base_train = spec.get("train", {})
    seq_len = base_train.get("seq_len", 128)

    records, done = [], set()
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            prev = json.load(open(checkpoint_path))
            records = [r for r in prev.get("records", [])
                       if math.isfinite(r.get("close_loss", float("nan")))]
            done = {(r.get("mode", "geodesic"), r["difficulty"], round(r["kappa"], 6), r["seed"])
                    for r in records}
            print(f"resuming: {len(done)} runs cached", flush=True)
        except Exception as e:
            print(f"could not resume ({e}); fresh", flush=True)

    total = len(modes) * len(diffs) * len(kappa_grid) * n_seeds
    t0, i = time.time(), 0
    for mode in modes:
        for d in diffs:
            data_eval = make_dataset("dyck", k=d["k"], max_depth=d["max_depth"], seed=99999)
            for kap in kappa_grid:
                for seed in range(n_seeds):
                    i += 1
                    key = (mode, d["name"], round(kap, 6), seed)
                    if key in done:
                        print(f"[{i}/{total}] {mode} {d['name']} k={kap:+.1f} s={seed} (cached)", flush=True)
                        continue
                    data = make_dataset("dyck", k=d["k"], max_depth=d["max_depth"], seed=seed)
                    mcfg = ModelConfig(vocab_size=data.vocab_size, kappa_init=kap,
                                       emb_scale=emb, attention_mode=mode, **base_model)
                    tcfg = TrainConfig(kappa_mode="fixed", seed=seed, **base_train)
                    res = run_training(mcfg, data, tcfg)
                    overall, close = eval_losses(res["model"], data_eval, seq_len)
                    rec = {"mode": mode, "difficulty": d["name"], "k": d["k"],
                           "max_depth": d["max_depth"], "kappa": kap, "seed": seed,
                           "overall_loss": overall, "close_loss": close,
                           "diverged": bool(res["diverged"])}
                    records.append(rec)
                    if checkpoint_path:
                        _atomic_dump({"spec": spec, "records": records}, checkpoint_path)
                    print(f"[{i}/{total}] {mode:8s} {d['name']:6s} k={kap:+.1f} s={seed} "
                          f"close={close:.3f} ({time.time()-t0:.0f}s)", flush=True)

    return {"spec": spec, "records": records, "wall_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/difficulty_results.json")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    spec = default_spec()
    if args.config and os.path.exists(args.config):
        spec = yaml.safe_load(open(args.config))
    if args.steps is not None:
        spec["train"]["steps"] = args.steps
    if args.device is not None:
        spec["train"]["device"] = args.device
    out = run_difficulty(spec, checkpoint_path=args.out, resume=not args.no_resume)
    _atomic_dump(out, args.out)
    print(f"\nwrote {args.out} ({len(out['records'])} runs)")


if __name__ == "__main__":
    main()
