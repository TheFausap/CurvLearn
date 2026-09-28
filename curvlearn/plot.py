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
    print("wrote phase diagram + trajectory plots to", args.outdir)


if __name__ == "__main__":
    main()
