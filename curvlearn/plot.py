"""Plot the Design A results: the flat-attractor phase diagram and the kappa trajectories.

    python -m curvlearn.plot --results results/designA_results.json --outdir results
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib.pyplot as plt


def _grids(out):
    k0 = out["spec"]["kappa0_grid"]
    emb = out["spec"]["emb_scale_grid"]
    return k0, emb


def phase_diagram(out, path):
    k0, emb = _grids(out)
    D = np.full((len(emb), len(k0)), np.nan)
    F = np.full((len(emb), len(k0)), np.nan)
    for c in out["cells"]:
        i, j = emb.index(c["emb_scale"]), k0.index(c["kappa_init"])
        D[i, j] = abs(c["delta"])
        F[i, j] = c["kappa_final"]
    fig, ax = plt.subplots(figsize=(1.3 * len(k0) + 2, 1.1 * len(emb) + 2))
    im = ax.imshow(D, cmap="magma", aspect="auto", origin="lower")
    ax.set_xticks(range(len(k0))); ax.set_xticklabels([f"{v:+.2g}" for v in k0])
    ax.set_yticks(range(len(emb))); ax.set_yticklabels([f"{v:g}" for v in emb])
    ax.set_xlabel(r"initial curvature $\kappa_0$")
    ax.set_ylabel("embedding scale")
    ax.set_title(r"Flat-attractor phase diagram: $|\kappa_{\rm final}-\kappa_0|$"
                 "\n(dark near $\\kappa_0=0$ = the curvature never moved)")
    for i in range(len(emb)):
        for j in range(len(k0)):
            ax.text(j, i, f"{F[i,j]:+.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$|\Delta\kappa|$ over training")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def loss_diagram(out, path):
    """Final training loss over the grid. Answers: does the curvature kappa settles at
    actually BUY anything, or is the loss flat in kappa (so the attractor is a shallow drift)?"""
    k0, emb = _grids(out)
    L = np.full((len(emb), len(k0)), np.nan)
    for c in out["cells"]:
        i, j = emb.index(c["emb_scale"]), k0.index(c["kappa_init"])
        L[i, j] = c["final_loss"]
    fig, ax = plt.subplots(figsize=(1.3 * len(k0) + 2, 1.1 * len(emb) + 2))
    im = ax.imshow(L, cmap="viridis_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(k0))); ax.set_xticklabels([f"{v:+.2g}" for v in k0])
    ax.set_yticks(range(len(emb))); ax.set_yticklabels([f"{v:g}" for v in emb])
    ax.set_xlabel(r"initial curvature $\kappa_0$"); ax.set_ylabel("embedding scale")
    ax.set_title("Final training loss (lower = better)\n"
                 "flat-in-$\\kappa$ => curvature is loss-neutral; a dip => a real optimum")
    for i in range(len(emb)):
        for j in range(len(k0)):
            if np.isfinite(L[i, j]):
                ax.text(j, i, f"{L[i,j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="final loss")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def trajectories(out, path):
    k0, emb = _grids(out)
    fig, axes = plt.subplots(1, len(emb), figsize=(4.2 * len(emb), 3.4), sharey=True)
    if len(emb) == 1:
        axes = [axes]
    cmap = plt.get_cmap("coolwarm")
    for ax, e in zip(axes, emb):
        for c in out["cells"]:
            if c["emb_scale"] != e:
                continue
            frac = (k0.index(c["kappa_init"]) + 0.5) / len(k0)
            ax.plot(c["step_traj"], c["kappa_traj"], color=cmap(frac), lw=1.5,
                    label=f"{c['kappa_init']:+.2g}")
        ax.axhline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"embedding scale = {e:g}")
        ax.set_xlabel("training step")
    axes[0].set_ylabel(r"$\kappa$")
    axes[-1].legend(title=r"$\kappa_0$", fontsize=7, ncol=2)
    fig.suptitle(r"Curvature trajectories: does $\kappa$ leave its initial value?")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def difficulty_plot(out, path):
    """Close-bracket loss vs fixed kappa, one line per difficulty. Left: absolute (with
    +/-1 std over seeds). Right: relative to kappa=0, the dip detector -- a curve that dives
    below 0 at kappa<0 as difficulty grows means hyperbolic curvature is finally buying
    something. Flat lines mean curvature is loss-neutral at that difficulty."""
    recs = out["records"]
    diffs = [d["name"] for d in out["spec"]["difficulties"]]
    kappas = sorted(set(r["kappa"] for r in recs))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4))
    cmap = plt.get_cmap("viridis")
    for di, name in enumerate(diffs):
        mean, std = [], []
        for kap in kappas:
            vals = [r["close_loss"] for r in recs
                    if r["difficulty"] == name and r["kappa"] == kap
                    and np.isfinite(r["close_loss"])]
            mean.append(np.mean(vals) if vals else np.nan)
            std.append(np.std(vals) if len(vals) > 1 else 0.0)
        mean, std = np.array(mean), np.array(std)
        col = cmap(0.15 + 0.7 * di / max(1, len(diffs) - 1))
        axL.plot(kappas, mean, "-o", color=col, label=name)
        axL.fill_between(kappas, mean - std, mean + std, color=col, alpha=0.2)
        z = kappas.index(0.0) if 0.0 in kappas else int(np.argmin(np.abs(kappas)))
        rel = mean - mean[z]
        axR.plot(kappas, rel, "-o", color=col, label=name)
    for ax in (axL, axR):
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel(r"fixed curvature $\kappa$")
    axR.axhline(0, color="k", lw=0.6, ls=":")
    axL.set_ylabel("close-bracket loss")
    axR.set_ylabel(r"close-loss relative to $\kappa=0$")
    axL.set_title("Absolute (mean $\\pm$ 1 std over seeds)")
    axR.set_title("Relative to flat: below 0 at $\\kappa<0$ = hyperbolic helps")
    axL.legend(title="difficulty"); axR.legend(title="difficulty")
    fig.suptitle("Difficulty sweep: does curvature become loss-relevant as hierarchy deepens?")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/designA_results.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    with open(args.results) as f:
        out = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)
    phase_diagram(out, os.path.join(args.outdir, "designA_phase_diagram.png"))
    trajectories(out, os.path.join(args.outdir, "designA_kappa_trajectories.png"))
    loss_diagram(out, os.path.join(args.outdir, "designA_loss.png"))
    print("wrote phase diagram + trajectory + loss plots to", args.outdir)


if __name__ == "__main__":
    main()
