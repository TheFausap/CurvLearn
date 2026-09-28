"""Design A: the flat-attractor phase diagram.

Sweep the INITIAL curvature kappa_0 against the embedding scale, train a small model with a
FREE (learnable) kappa in each cell, and record where kappa ended up. The prediction (memo
Sec 2): a dead basin around kappa_0 = 0 where kappa never moves, escaping only past a
|kappa_0| threshold that shrinks as the embedding scale grows.

Usage:
    python -m curvlearn.sweep --config experiments/designA.yaml
    python -m curvlearn.sweep --steps 1500 --device cuda --out results/designA_results.json
"""
from __future__ import annotations
import argparse, json, math, os, time
import yaml

from .model import ModelConfig
from .train import TrainConfig, run_training
from .data import make_dataset


def _atomic_dump(obj, path):
    """Write JSON to ``path`` via a temp file + rename, so a killed runtime never leaves a
    half-written checkpoint (important on Colab, where the kernel can vanish mid-write)."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def run_sweep(spec: dict, checkpoint_path: str | None = None, resume: bool = True):
    """Run the Design A sweep.

    If ``checkpoint_path`` is given, the accumulated results are written there after *every*
    cell (atomically), and -- when ``resume`` is True -- cells already present in that file
    are skipped. So a Colab disconnect costs at most one cell, and re-running the notebook
    picks up where it stopped. Point ``checkpoint_path`` at Google Drive for persistence.
    """
    ds = spec["dataset"]
    data = make_dataset(ds["name"], **{k: v for k, v in ds.items() if k != "name"})
    vocab = data.vocab_size

    kappa0_grid = spec["kappa0_grid"]
    emb_grid = spec["emb_scale_grid"]
    base_model = {k: v for k, v in spec.get("model", {}).items() if k != "seq_len"}
    base_train = spec.get("train", {})

    cells, done = [], set()
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            prev = json.load(open(checkpoint_path))
            # Keep only cells that actually SUCCEEDED. A diverged / NaN cell is not "done" --
            # it must be retried (e.g. after a code fix), so drop it and let it recompute.
            good = [c for c in prev.get("cells", [])
                    if math.isfinite(c.get("kappa_final", float("nan")))
                    and not c.get("diverged", False)]
            dropped = len(prev.get("cells", [])) - len(good)
            cells = good
            done = {(round(c["kappa_init"], 6), round(c["emb_scale"], 6)) for c in good}
            print(f"resuming: {len(done)} successful cells cached, "
                  f"{dropped} failed/NaN cells will be recomputed", flush=True)
        except Exception as e:
            print(f"could not resume ({e}); starting fresh", flush=True)

    t0 = time.time()
    total = len(kappa0_grid) * len(emb_grid)
    i = 0
    for emb in emb_grid:
        for k0 in kappa0_grid:
            i += 1
            if (round(k0, 6), round(emb, 6)) in done:
                print(f"[{i}/{total}] emb={emb:>4} k0={k0:+.2f} -> (cached)", flush=True)
                continue
            mcfg = ModelConfig(vocab_size=vocab, kappa_init=k0, emb_scale=emb, **base_model)
            tcfg = TrainConfig(kappa_mode="learn", **base_train)
            res = run_training(mcfg, data, tcfg)
            cell = {"kappa_init": k0, "emb_scale": emb,
                    "kappa_final": res["kappa_final"],
                    "delta": res["kappa_final"] - k0,
                    "final_loss": res["final_loss"],
                    "kappa_traj": res["history"]["kappa"],
                    "kappa_grad_traj": res["history"]["kappa_grad"],
                    "step_traj": res["history"]["step"]}
            cells.append(cell)
            out = {"spec": spec, "cells": cells, "vocab_size": vocab,
                   "wall_seconds": time.time() - t0}
            if checkpoint_path:
                _atomic_dump(out, checkpoint_path)
            print(f"[{i}/{total}] emb={emb:>4} k0={k0:+.2f} -> k_final={cell['kappa_final']:+.3f} "
                  f"|delta|={abs(cell['delta']):.3f} loss={cell['final_loss']:.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    return {"spec": spec, "cells": cells, "vocab_size": vocab,
            "wall_seconds": time.time() - t0}


def default_spec():
    return {
        "dataset": {"name": "dyck", "k": 3, "max_depth": 12},
        "kappa0_grid": [-2.0, -1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 2.0],
        "emb_scale_grid": [0.3, 1.0, 3.0],
        "model": {"d_model": 64, "n_layers": 2, "n_heads": 1, "seq_len": 128,
                  "geodesic_output": True},
        "train": {"steps": 2000, "batch_size": 64, "seq_len": 128,
                  "lr": 3e-3, "kappa_lr": 5e-2, "device": "cuda", "log_every": 25},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default="results/designA_results.json")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing checkpoint at --out and start fresh")
    args = ap.parse_args()

    spec = default_spec()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            spec = yaml.safe_load(f)
    # model config keys that belong to ModelConfig (strip seq_len which is a train arg)
    spec["model"].pop("seq_len", None)
    if args.steps is not None:
        spec["train"]["steps"] = args.steps
    if args.device is not None:
        spec["train"]["device"] = args.device

    # checkpoint == output path: incremental, atomic, and resumable across restarts
    out = run_sweep(spec, checkpoint_path=args.out, resume=not args.no_resume)
    _atomic_dump(out, args.out)
    print(f"\nwrote {args.out}  ({len(out['cells'])} cells, {out['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
