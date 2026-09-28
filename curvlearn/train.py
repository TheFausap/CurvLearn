"""Single training run for a GeometricAttentionLM, instrumented for curvature dynamics.

The whole point of Design A is to watch kappa. So we:
  * put kappa in its own optimizer param-group with a larger LR (its Euclidean gradient is
    tiny near flat -- see the memo; a shared LR guarantees it never moves);
  * log kappa, its gradient, and the loss on a fixed cadence;
  * support three curvature modes -- "learn" (free parameter, Design A), "fixed" (frozen),
    and "schedule" (kappa(step) supplied by a callable, the hook Design B will use).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional
import torch

from .model import GeometricAttentionLM, ModelConfig


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 64
    seq_len: int = 128
    lr: float = 3e-3
    kappa_lr: float = 5e-2          # deliberately >> lr; the flat-attractor conditioning fix
    kappa_mode: str = "learn"       # "learn" | "fixed" | "schedule"
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 25
    device: str = "cuda"
    seed: int = 0


def run_training(mcfg: ModelConfig, data, tcfg: TrainConfig,
                 kappa_schedule: Optional[Callable[[int], float]] = None):
    torch.manual_seed(tcfg.seed)
    dev = tcfg.device if torch.cuda.is_available() or tcfg.device == "cpu" else "cpu"
    model = GeometricAttentionLM(mcfg).to(dev)

    if tcfg.kappa_mode == "fixed":
        model.kappa.requires_grad_(False)

    kappa_params = [model.kappa]
    other = [p for n, p in model.named_parameters() if n != "kappa"]
    groups = [{"params": other, "lr": tcfg.lr, "weight_decay": tcfg.weight_decay}]
    if tcfg.kappa_mode == "learn":
        groups.append({"params": kappa_params, "lr": tcfg.kappa_lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)

    hist = {"step": [], "loss": [], "kappa": [], "kappa_grad": []}
    for step in range(tcfg.steps + 1):
        if tcfg.kappa_mode == "schedule" and kappa_schedule is not None:
            with torch.no_grad():
                model.kappa.fill_(float(kappa_schedule(step)))

        x, y = data.batch(tcfg.batch_size, tcfg.seq_len, device=dev)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            print(f"  [step {step}] non-finite loss ({float(loss.detach())}); stopping this run. "
                  f"Lower emb_scale / lr, or check the |kappa| range.", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        kg = float(model.kappa.grad.detach()) if model.kappa.grad is not None else 0.0
        if tcfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()

        if step % tcfg.log_every == 0:
            hist["step"].append(step)
            hist["loss"].append(float(loss.detach()))
            hist["kappa"].append(float(model.kappa.detach()))
            hist["kappa_grad"].append(kg)

    final_loss = hist["loss"][-1] if hist["loss"] else float("nan")
    return {"model": model, "history": hist,
            "kappa_init": float(mcfg.kappa_init), "kappa_final": float(model.kappa.detach()),
            "emb_scale": float(mcfg.emb_scale), "final_loss": final_loss,
            "diverged": not (hist["loss"] and step >= tcfg.steps)}
