"""Per-subspace mixed curvature: does letting each attention head choose its own curvature beat
a single shared curvature?

Product manifold = one kappa_stereographic factor per head, curvatures learned independently
(ProductGyroLM). Compare, in the crossover regime the dimension sweep found:

  shared    n_curv_factors = 1        (one learnable kappa for all heads -- the control)
  per_head  n_curv_factors = n_heads  (a distinct learnable kappa per head -- mixed curvature)

Reads: does per_head lower close-bracket loss vs shared, and do the heads SPECIALISE to
different curvatures (the learned kappa vector spreading out)?
"""
from __future__ import annotations
import argparse, json, math, os, time
import yaml
import torch

from .model import ModelConfig, ProductGyroLM
from .data import make_dataset
from .difficulty import eval_losses, _atomic_dump


def train_one(vocab, data, dim, n_heads, nf, emb_scale, steps, bs, seq_len,
              lr, kappa_lr, seed, device, kappa_init_common=None):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=vocab, d_model=dim, n_heads=n_heads, n_layers=2,
                      emb_scale=emb_scale, max_len=max(256, seq_len),
                      geodesic_output=True)   # curvature-coupled readout (fixes the linear-head confound)
    model = ProductGyroLM(cfg, n_curv_factors=nf, geodesic_output=True,
                          kappa_init_common=kappa_init_common).to(device)
    kappa_params = [model.kappa]
    others = [p for n, p in model.named_parameters() if n != "kappa"]
    opt = torch.optim.AdamW([{"params": others, "lr": lr},
                             {"params": kappa_params, "lr": kappa_lr, "weight_decay": 0.0}])
    ktraj, steps_log = [], []
    for step in range(steps + 1):
        x, y = data.batch(bs, seq_len, device=device)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            print(f"    non-finite loss at step {step}; stopping", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            ktraj.append(model._k().detach().cpu().tolist())
            steps_log.append(step)
    return model, {"kappa_final": model._k().detach().cpu().tolist(),
                   "kappa_traj": ktraj, "step_traj": steps_log}


def default_spec():
    return {
        "regimes": [
            {"name": "medium", "k": 3, "max_depth": 25, "dim": 4, "n_heads": 2},
            {"name": "hard",   "k": 5, "max_depth": 50, "dim": 8, "n_heads": 4},
        ],
        "factors": ["shared", "per_head"],
        "seeds": 3,
        "emb_scale": 1.0,
        # kappa_lr lowered from 5e-2: with the geodesic readout kappa now gets a strong
        # likelihood gradient, and 5e-2 overshot to the +-4 clamp in the linear-head run.
        "train": {"steps": 2000, "batch_size": 64, "seq_len": 128,
                  "lr": 3e-3, "kappa_lr": 2e-2, "device": "cuda"},
    }


def run_product_sweep(spec, checkpoint_path=None, resume=True):
    regimes = spec["regimes"]
    factors = spec["factors"]
    n_seeds = spec["seeds"]
    emb = spec.get("emb_scale", 1.0)
    kic = spec.get("kappa_init_common", None)   # None -> spread (linspace); float -> common init control
    tr = spec["train"]
    dev = tr.get("device", "cuda")
    dev = dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu"

    records, done = [], set()
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            prev = json.load(open(checkpoint_path))
            records = [r for r in prev.get("records", [])
                       if math.isfinite(r.get("close_loss", float("nan")))]
            done = {(r["regime"], r["factor"], r["seed"]) for r in records}
            print(f"resuming: {len(done)} runs cached", flush=True)
        except Exception as e:
            print(f"could not resume ({e}); fresh", flush=True)

    total = len(regimes) * len(factors) * n_seeds
    t0, i = time.time(), 0
    for reg in regimes:
        data_eval = make_dataset("dyck", k=reg["k"], max_depth=reg["max_depth"], seed=99999)
        for factor in factors:
            nf = 1 if factor == "shared" else reg["n_heads"]
            for seed in range(n_seeds):
                i += 1
                key = (reg["name"], factor, seed)
                if key in done:
                    print(f"[{i}/{total}] {reg['name']} {factor} s={seed} (cached)", flush=True)
                    continue
                data = make_dataset("dyck", k=reg["k"], max_depth=reg["max_depth"], seed=seed)
                model, kinfo = train_one(data.vocab_size, data, reg["dim"], reg["n_heads"], nf,
                                         emb, tr["steps"], tr["batch_size"], tr["seq_len"],
                                         tr["lr"], tr["kappa_lr"], seed, dev,
                                         kappa_init_common=kic)
                overall, close = eval_losses(model, data_eval, tr["seq_len"])
                rec = {"regime": reg["name"], "dim": reg["dim"], "n_heads": reg["n_heads"],
                       "factor": factor, "seed": seed, "close_loss": close,
                       "kappa_init_common": kic,
                       "overall_loss": overall, "kappa_final": kinfo["kappa_final"],
                       "kappa_traj": kinfo["kappa_traj"], "step_traj": kinfo["step_traj"]}
                records.append(rec)
                if checkpoint_path:
                    _atomic_dump({"spec": spec, "records": records}, checkpoint_path)
                print(f"[{i}/{total}] {reg['name']:6s} {factor:8s} s={seed} close={close:.3f} "
                      f"k={['%.2f'%v for v in kinfo['kappa_final']]} ({time.time()-t0:.0f}s)", flush=True)

    return {"spec": spec, "records": records, "wall_seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/product_results.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    spec = default_spec()
    if args.config and os.path.exists(args.config):
        spec = yaml.safe_load(open(args.config))
    if args.device is not None:
        spec["train"]["device"] = args.device
    out = run_product_sweep(spec, checkpoint_path=args.out, resume=not args.no_resume)
    _atomic_dump(out, args.out)
    print(f"\nwrote {args.out} ({len(out['records'])} runs)")


if __name__ == "__main__":
    main()
