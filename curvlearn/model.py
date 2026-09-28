"""GeometricAttentionLM -- a minimal transformer whose representation space has a single
learnable / schedulable curvature ``kappa``.

Design choices are deliberately the *simplest ones that couple kappa to the loss*, because
Design A studies whether kappa moves, not raw perplexity:

  * tokens are embedded as tangent vectors at the origin, scaled by ``emb_scale`` (the sweep
    knob), then mapped onto the manifold with ``expmap0``;
  * attention scores are ``-beta * d_kappa(x_i, x_j)^2`` (a geodesic-distance / kernel
    attention), so curvature enters the mixing directly;
  * the output head is *geodesic decoding*: ``logit_v = -beta_out * d_kappa(x, o_v)^2`` to a
    table of vocabulary points, so curvature enters the likelihood as well.

Everything geometric runs in float32. ``kappa`` is an ordinary Euclidean nn.Parameter (a
scalar): we want to watch its *Euclidean* gradient, so do NOT put it on a Riemannian
optimizer -- that would change the very dynamics under study.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as G


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 1
    max_len: int = 256
    kappa_init: float = -1.0        # NON-flat by default: flat is an attractor (see memo Sec 2)
    emb_scale: float = 1.0          # swept in Design A
    ffn_mult: int = 2
    geodesic_output: bool = True    # False -> plain linear head (kappa only in attention)
    norm_features: bool = False     # LayerNorm shrinks norms and fights curvature; off by default
    emb_init_std: float = 0.02      # small init: tokens must start INSIDE the ball, not on it
    attention_mode: str = "geodesic"  # "geodesic": scores = -beta*d_kappa^2 (Design A/A').
                                       # "gyro": tangent-space Q.K scores + hyperbolic
                                       #   gyromidpoint aggregation ("place, don't score").


class GeoAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.mode = cfg.attention_mode
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.Wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.Wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if self.mode == "gyro":
            self.Wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.Wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, k, causal_mask):
        B, T, D = x.shape
        xt = G.logmap0(x, k)                                   # tangent features (B,T,D)
        xh = x.view(B, T, self.h, self.dh).transpose(1, 2)     # (B,H,T,dh) product-of-balls

        if self.mode == "geodesic":
            scores = -self.beta.abs() * G.pairwise_dist2(xh, k)          # curvature in the SCORES
            A = torch.softmax(scores.masked_fill(causal_mask, float("-inf")), dim=-1)
            vh = self.Wv(xt).view(B, T, self.h, self.dh).transpose(1, 2)
            agg = torch.matmul(A, vh).transpose(1, 2).reshape(B, T, D)   # Euclidean tangent sum
            return xt + self.Wo(agg)

        # "gyro": tangent-space Q.K scores (curvature NOT in the score), curvature enters via a
        # hyperbolic gyromidpoint aggregation of the value POINTS -- "place, don't score".
        q = self.Wq(xt).view(B, T, self.h, self.dh).transpose(1, 2)      # (B,H,T,dh)
        kk_ = self.Wk(xt).view(B, T, self.h, self.dh).transpose(1, 2)
        scores = torch.matmul(q, kk_.transpose(-2, -1)) / (self.dh ** 0.5)
        A = torch.softmax(scores.masked_fill(causal_mask, float("-inf")), dim=-1)  # (B,H,T,T)
        agg_pts = G.weighted_midpoint(xh, A, k)                          # (B,H,T,dh) on manifold
        agg = G.logmap0(agg_pts, k).transpose(1, 2).reshape(B, T, D)     # back to tangent
        return xt + self.Wo(agg)


class TangentFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.ffn_mult
        self.net = nn.Sequential(nn.Linear(cfg.d_model, hidden), nn.GELU(),
                                 nn.Linear(hidden, cfg.d_model))

    def forward(self, xt):
        return xt + self.net(xt)


class GeometricAttentionLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.kappa = nn.Parameter(torch.tensor(float(cfg.kappa_init)))
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({"attn": GeoAttention(cfg), "ffn": TangentFFN(cfg)}))
        self.norm = nn.LayerNorm(cfg.d_model) if cfg.norm_features else nn.Identity()
        if cfg.geodesic_output:
            self.out_pts = nn.Embedding(cfg.vocab_size, cfg.d_model)   # vocab tangent points
            self.beta_out = nn.Parameter(torch.tensor(1.0))
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)

        # Small init so exp_0(embedding) lands well inside the ball. Default N(0,1) gives a
        # tangent of norm ~sqrt(d_model), which exp_0 pushes onto the boundary (kappa<0) where
        # the geometry is numerically singular -> NaNs. emb_scale then dials the interior
        # radius up from here (and is the Design-A sweep knob).
        s = cfg.emb_init_std
        nn.init.normal_(self.tok.weight, std=s)
        nn.init.normal_(self.pos.weight, std=s)
        if cfg.geodesic_output:
            nn.init.normal_(self.out_pts.weight, std=s)

    def _k(self):
        # keep curvature in a safe band; f32 for all geometry
        return self.kappa.float().clamp(-4.0, 4.0).view(())

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        k = self._k()
        pos = torch.arange(T, device=tokens.device)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            t = (self.tok(tokens) + self.pos(pos)[None]) * self.cfg.emb_scale
            x = G.expmap0(t, k)
            cmask = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), 1)[None, None]
            for layer in self.layers:
                xt = layer["attn"](x, k, cmask)
                xt = layer["ffn"](xt)
                x = G.expmap0(xt, k)
            h = self.norm(G.logmap0(x, k))
            if self.cfg.geodesic_output:
                xf = G.expmap0(h, k)                                    # (B,T,D)
                o = G.expmap0(self.out_pts.weight * self.cfg.emb_scale, k)   # (V,D)
                d2 = G.dist(xf.unsqueeze(-2), o[None, None], k) ** 2         # (B,T,V)
                logits = -self.beta_out.abs() * d2
            else:
                logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class ProductGyroAttention(nn.Module):
    """Gyro attention on a PRODUCT manifold: each head h has its own curvature kappa[h].
    Curvature enters only through the per-head hyperbolic gyromidpoint aggregation; scores are
    Euclidean tangent-space Q.K. Returns the attention delta (residual added by the model)."""

    def __init__(self, cfg: "ModelConfig", n_factors: int):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.nf = n_factors
        self.Wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.Wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.Wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, t, kappa, causal_mask):
        B, T, D = t.shape
        q = self.Wq(t).view(B, T, self.h, self.dh).transpose(1, 2)   # (B,h,T,dh)
        k = self.Wk(t).view(B, T, self.h, self.dh).transpose(1, 2)
        th = t.view(B, T, self.h, self.dh).transpose(1, 2)           # (B,h,T,dh) tangent per head
        outs = []
        for hd in range(self.h):
            kap = kappa[0] if self.nf == 1 else kappa[hd]            # scalar tensor, keeps grad
            s = torch.matmul(q[:, hd], k[:, hd].transpose(-2, -1)) / (self.dh ** 0.5)  # (B,T,T)
            A = torch.softmax(s.masked_fill(causal_mask, float("-inf")), dim=-1)
            pts = G.expmap0(th[:, hd], kap)                          # (B,T,dh) manifold points
            agg = G.logmap0(G.weighted_midpoint(pts, A, kap), kap)   # (B,T,dh) back to tangent
            outs.append(agg)
        return self.Wo(torch.cat(outs, dim=-1))                      # (B,T,D) delta


class ProductGyroLM(nn.Module):
    """Tangent-trunk transformer with per-head (product-manifold) curvature.

    The trunk carries tangent features (the layer-boundary exp/log maps of the single-kappa
    model cancel, so they are skipped here); curvature lives only in the per-head gyromidpoint
    aggregation. Set n_curv_factors=1 for a single SHARED learnable curvature (the control), or
    =n_heads for a distinct learnable curvature per head (mixed curvature).

    Readout: with geodesic_output=True (default via cfg) the head is a per-head curvature-coupled
    geodesic decode -- each head's tangent slice is decoded against its slice of the vocab points
    using that head's kappa, and the per-head squared distances are summed. This couples every
    kappa to the likelihood (as Design B's head did), fixing the earlier linear-head confound
    where kappa got only a weak aggregation gradient and drifted to the clamp. geodesic_output=False
    restores the plain linear head.
    """

    def __init__(self, cfg: "ModelConfig", n_curv_factors=None, geodesic_output=None,
                 kappa_init_common=None):
        super().__init__()
        self.cfg = cfg
        nf = int(n_curv_factors or cfg.n_heads)
        assert nf in (1, cfg.n_heads), "n_curv_factors must be 1 (shared) or n_heads (per-head)"
        self.nf = nf
        self.h = cfg.n_heads
        self.dh = cfg.d_model // cfg.n_heads
        # Curvature-coupled readout (Design B style) restores the kappa gradient a plain linear
        # head severs. On the product manifold the decode is per-head: each head's tangent slice
        # is decoded against its slice of the vocab points with that head's kappa, and the
        # per-head squared geodesic distances are summed (the product-manifold distance^2). For
        # nf==1 this reduces exactly to the single-curvature geodesic head of GeometricAttentionLM.
        self.geodesic_output = bool(cfg.geodesic_output if geodesic_output is None else geodesic_output)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        # Default init spreads the heads (linspace) -- but that lets a weak gradient merely
        # preserve init ORDER. kappa_init_common initialises every head at the SAME value, so
        # any spread that emerges is gradient-driven specialisation, not inherited order
        # (the decisive control for "do heads specialise?").
        if kappa_init_common is not None:
            init = torch.full((nf,), float(kappa_init_common))
        else:
            init = torch.tensor([-1.0]) if nf == 1 else torch.linspace(-1.5, -0.2, nf)
        self.kappa = nn.Parameter(init)                              # per-factor curvature
        self.layers = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(nn.ModuleDict({
                "attn": ProductGyroAttention(cfg, nf),
                "ffn": nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model * cfg.ffn_mult),
                                     nn.GELU(),
                                     nn.Linear(cfg.d_model * cfg.ffn_mult, cfg.d_model)),
            }))
        self.norm = nn.LayerNorm(cfg.d_model)
        s = cfg.emb_init_std
        if self.geodesic_output:
            self.out_pts = nn.Embedding(cfg.vocab_size, cfg.d_model)   # vocab tangent points
            self.beta_out = nn.Parameter(torch.tensor(1.0))
            nn.init.normal_(self.out_pts.weight, std=s)
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        nn.init.normal_(self.tok.weight, std=s)
        nn.init.normal_(self.pos.weight, std=s)

    def _k(self):
        return self.kappa.float().clamp(-4.0, 4.0)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        kappa = self._k()
        pos = torch.arange(T, device=tokens.device)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            t = (self.tok(tokens) + self.pos(pos)[None]) * self.cfg.emb_scale   # tangent
            cmask = torch.triu(torch.ones(T, T, device=tokens.device, dtype=torch.bool), 1)
            for layer in self.layers:
                t = t + layer["attn"](t, kappa, cmask)
                t = t + layer["ffn"](t)
            hh = self.norm(t)                                        # (B,T,D) tangent
            if self.geodesic_output:
                hv = hh.view(B, T, self.h, self.dh)                          # (B,T,h,dh)
                ov = (self.out_pts.weight * self.cfg.emb_scale).view(
                    self.cfg.vocab_size, self.h, self.dh)                    # (V,h,dh)
                d2 = 0.0
                for hd in range(self.h):
                    kap = kappa[0] if self.nf == 1 else kappa[hd]
                    xf = G.expmap0(hv[:, :, hd], kap)                        # (B,T,dh)
                    o = G.expmap0(ov[:, hd], kap)                            # (V,dh)
                    d2 = d2 + G.dist(xf.unsqueeze(-2), o[None, None], kap) ** 2  # (B,T,V)
                logits = -self.beta_out.abs() * d2
            else:
                logits = self.head(hh)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
