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
import argparse, json, os, time
import yaml

from .model import ModelConfig
from .train import TrainConfig, run_training
from .data import make_dataset


def run_sweep(spec: dict):
    ds = spec["dataset"]
    data = make_dataset(ds["name"], **{k: v for k, v in ds.items() if k != "name"})
    vocab = data.vocab_size

    kappa0_grid = spec["kappa0_grid"]
    emb_grid = spec["emb_scale_grid"]
    base_model = {k: v for k, v in spec.get("model", {}).items() if k != "seq_len"}
    base_train = spec.get("train", {})

    cells = []
    t0 = time.time()
    total = len(kappa0_grid) * len(emb_grid)
    i = 0
    for emb in emb_grid:
        for k0 in kappa0_grid:
            i += 1
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

    out = run_sweep(spec)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {args.out}  ({len(out['cells'])} cells, {out['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
