"""Design B -- can the model LEARN to use the beneficial curvature, and does a schedule beat
free learning?

Run in the regime the dimension sweep proved has a real curvature optimum (gyro aggregation,
small d_model matched to task: medium@d=4, hard@d=8, optimum kappa ~ -0.5..-1). Five settings
per regime, over seeds:

  flat            fixed kappa=0                       (Euclidean baseline)
  fixed_opt       fixed kappa=-0.5                    (oracle: the known-good curvature)
  learn_from_opt  learnable kappa, non-flat init -1   (does it stay in the good basin?)
  learn_from_flat learnable kappa, flat init 0        (does the flat attractor trap it?)
  anneal          scheduled kappa 0 -> -1             (curriculum: can a schedule escape flat?)

The Design-A payoff test: if learn_from_flat stalls near 0 (higher loss) while anneal reaches
-1 (loss matching fixed_opt), then scheduling beats free learning precisely because it escapes
the flat attractor -- the original "start non-flat / tune during training" hypothesis, now in a
regime where there is actually a better curvature to reach.
"""
from __future__ import annotations
import argparse, json, math, os, time
import yaml

from .model import ModelConfig
from .train import TrainConfig, run_training
from .data import make_dataset
from .difficulty import eval_losses, _atomic_dump


def make_linear_sched(a, b, frac, steps):
    T = max(1, int(frac * steps))
    return lambda step: a + (b - a) * min(1.0, step / T)


def settings(steps):
    return [
        {"name": "flat",            "mode": "fixed",    "kappa_init": 0.0},
        {"name": "fixed_opt",       "mode": "fixed",    "kappa_init": -0.5},
        {"name": "learn_from_opt",  "mode": "learn",    "kappa_init": -1.0},
        {"name": "learn_from_flat", "mode": "learn",    "kappa_init": 0.0},
        {"name": "anneal",          "mode": "schedule", "kappa_init": 0.0,
         "sched": (0.0, -1.0, 0.5)},   # linear 0 -> -1 over first 50% of training, then hold
    ]


def default_spec():
    return {
        "regimes": [
            {"name": "medium", "k": 3, "max_depth": 25, "dim": 4},
            {"name": "hard",   "k": 5, "max_depth": 50, "dim": 8},
        ],
        "seeds": 3,
        "emb_scale": 1.0,
        "attention_mode": "gyro",
        "model": {"n_layers": 2, "n_heads": 1, "geodesic_output": True},
        "train": {"steps": 2000, "batch_size": 64, "seq_len": 128,
                  "lr": 3e-3, "kappa_lr": 5e-2, "device": "cuda", "log_every": 50},
    }


def run_design_b(spec, checkpoint_path=None, resume=True):
    regimes = spec["regimes"]
    n_seeds = spec["seeds"]
    emb = spec.get("emb_scale", 1.0)
    mode = spec.get("attention_mode", "gyro")
    base_model = {k: v for k, v in spec.get("model", {}).items()
                  if k not in ("seq_len", "attention_mode", "d_model", "kappa_init")}
    base_train = spec.get("train", {})
    seq_len = base_train.get("seq_len", 128)
    steps = base_train.get("steps", 2000)
    setts = settings(steps)

    records, done = [], set()
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            prev = json.load(open(checkpoint_path))
            records = [r for r in prev.get("records", [])
                       if math.isfinite(r.get("close_loss", float("nan")))]
            done = {(r["regime"], r["setting"], r["seed"]) for r in records}
            print(f"resuming: {len(done)} runs cached", flush=True)
        except Exception as e:
            print(f"could not resume ({e}); fresh", flush=True)

    total = len(regimes) * len(setts) * n_seeds
    t0, i = time.time(), 0
    for reg in regimes:
        data_eval = make_dataset("dyck", k=reg["k"], max_depth=reg["max_depth"], seed=99999)
        for st in setts:
            for seed in range(n_seeds):
                i += 1
                key = (reg["name"], st["name"], seed)
                if key in done:
                    print(f"[{i}/{total}] {reg['name']} {st['name']} s={seed} (cached)", flush=True)
                    continue
                data = make_dataset("dyck", k=reg["k"], max_depth=reg["max_depth"], seed=seed)
                mcfg = ModelConfig(vocab_size=data.vocab_size, kappa_init=st["kappa_init"],
                                   emb_scale=emb, attention_mode=mode, d_model=reg["dim"],
                                   **base_model)
                tcfg = TrainConfig(kappa_mode=st["mode"], seed=seed, **base_train)
                sched = None
                if st["mode"] == "schedule":
                    a, b, frac = st["sched"]
                    sched = make_linear_sched(a, b, frac, steps)
                res = run_training(mcfg, data, tcfg, kappa_schedule=sched)
                overall, close = eval_losses(res["model"], data_eval, seq_len)
                rec = {"regime": reg["name"], "dim": reg["dim"], "setting": st["name"],
                       "seed": seed, "close_loss": close, "overall_loss": overall,
                       "kappa_final": res["kappa_final"], "kappa_traj": res["history"]["kappa"],
                       "step_traj": res["history"]["step"], "diverged": bool(res["diverged"])}
                records.append(rec)
                if checkpoint_path:
                    _atomic_dump({"spec": spec, "records": records}, checkpoint_path)
                print(f"[{i}/{total}] {reg['name']:6s} {st['name']:16s} s={seed} "
                      f"close={close:.3f} k_final={res['kappa_final']:+.2f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    return {"spec": spec, "records": records, "wall_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/design_b_results.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    spec = default_spec()
    if args.config and os.path.exists(args.config):
        spec = yaml.safe_load(open(args.config))
    if args.device is not None:
        spec["train"]["device"] = args.device
    out = run_design_b(spec, checkpoint_path=args.out, resume=not args.no_resume)
    _atomic_dump(out, args.out)
    print(f"\nwrote {args.out} ({len(out['records'])} runs)")


if __name__ == "__main__":
    main()
