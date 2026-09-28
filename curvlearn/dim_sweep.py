"""Dimension sweep: does hyperbolic aggregation beat flat once the dimension is a binding
constraint?

A'' showed hyperbolic's tree-embedding advantage is largest at LOW dimension; the mechanism
comparison found gyro halves the hard-task penalty at d_model=32 but doesn't cross below flat.
The hypothesis (and the user's prior experience -- "too much space goes flat"): the crossover
appears only when d_model is small enough that Euclidean space can no longer pack the hierarchy.

This sweeps d_model in {4,8,16,32} for the gyro mechanism on medium+hard Dyck, fixed kappa, over
seeds. Read plot.dim_sweep_plot: the "hyperbolic advantage" (best kappa<0 minus kappa=0) should
go NEGATIVE (hyperbolic beats flat) as d_model shrinks, first and most on the hard task.
"""
from __future__ import annotations
import argparse, json, math, os, time
import yaml

from .model import ModelConfig
from .train import TrainConfig, run_training
from .data import make_dataset
from .difficulty import eval_losses, _atomic_dump


def default_spec():
    return {
        "difficulties": [
            {"name": "medium", "k": 3, "max_depth": 25},
            {"name": "hard",   "k": 5, "max_depth": 50},
        ],
        "dims": [4, 8, 16, 32],
        "kappa_grid": [-2.0, -1.0, -0.5, 0.0],
        "seeds": 3,
        "emb_scale": 1.0,
        "attention_mode": "gyro",
        "model": {"n_layers": 2, "n_heads": 1, "geodesic_output": True},
        "train": {"steps": 2000, "batch_size": 64, "seq_len": 128,
                  "lr": 3e-3, "device": "cuda", "log_every": 50},
    }


def run_dim_sweep(spec, checkpoint_path=None, resume=True):
    diffs = spec["difficulties"]
    dims = spec["dims"]
    kappa_grid = spec["kappa_grid"]
    n_seeds = spec["seeds"]
    emb = spec.get("emb_scale", 1.0)
    mode = spec.get("attention_mode", "gyro")
    base_model = {k: v for k, v in spec.get("model", {}).items()
                  if k not in ("seq_len", "attention_mode", "d_model")}
    base_train = spec.get("train", {})
    seq_len = base_train.get("seq_len", 128)

    records, done = [], set()
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            prev = json.load(open(checkpoint_path))
            records = [r for r in prev.get("records", [])
                       if math.isfinite(r.get("close_loss", float("nan")))]
            done = {(r["dim"], r["difficulty"], round(r["kappa"], 6), r["seed"]) for r in records}
            print(f"resuming: {len(done)} runs cached", flush=True)
        except Exception as e:
            print(f"could not resume ({e}); fresh", flush=True)

    total = len(dims) * len(diffs) * len(kappa_grid) * n_seeds
    t0, i = time.time(), 0
    for dim in dims:
        for d in diffs:
            data_eval = make_dataset("dyck", k=d["k"], max_depth=d["max_depth"], seed=99999)
            for kap in kappa_grid:
                for seed in range(n_seeds):
                    i += 1
                    key = (dim, d["name"], round(kap, 6), seed)
                    if key in done:
                        print(f"[{i}/{total}] d={dim} {d['name']} k={kap:+.1f} s={seed} (cached)", flush=True)
                        continue
                    data = make_dataset("dyck", k=d["k"], max_depth=d["max_depth"], seed=seed)
                    mcfg = ModelConfig(vocab_size=data.vocab_size, kappa_init=kap, emb_scale=emb,
                                       attention_mode=mode, d_model=dim, **base_model)
                    tcfg = TrainConfig(kappa_mode="fixed", seed=seed, **base_train)
                    res = run_training(mcfg, data, tcfg)
                    overall, close = eval_losses(res["model"], data_eval, seq_len)
                    rec = {"dim": dim, "difficulty": d["name"], "k": d["k"],
                           "max_depth": d["max_depth"], "kappa": kap, "seed": seed,
                           "mode": mode, "overall_loss": overall, "close_loss": close,
                           "diverged": bool(res["diverged"])}
                    records.append(rec)
                    if checkpoint_path:
                        _atomic_dump({"spec": spec, "records": records}, checkpoint_path)
                    print(f"[{i}/{total}] d={dim:<2} {d['name']:6s} k={kap:+.1f} s={seed} "
                          f"close={close:.3f} ({time.time()-t0:.0f}s)", flush=True)

    return {"spec": spec, "records": records, "wall_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/dim_sweep_results.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    spec = default_spec()
    if args.config and os.path.exists(args.config):
        spec = yaml.safe_load(open(args.config))
    if args.device is not None:
        spec["train"]["device"] = args.device
    out = run_dim_sweep(spec, checkpoint_path=args.out, resume=not args.no_resume)
    _atomic_dump(out, args.out)
    print(f"\nwrote {args.out} ({len(out['records'])} runs)")


if __name__ == "__main__":
    main()
