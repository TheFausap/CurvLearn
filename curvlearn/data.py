"""Data for Design A.

Default: **Dyck-k**, balanced nested brackets of k types -- the canonical hierarchy task.
Next-token prediction forces the model to track an unbounded stack, so a hierarchical
(hyperbolic) representation has something to buy. It is synthetic, needs no download, and is
fast enough to run the whole sweep on CPU if necessary.

Also provided: a char-level loader for a local text file (enwik8 / tiny-shakespeare / code),
for when you want a realistic corpus on the A100.
"""
from __future__ import annotations
import random
import torch


class DyckK:
    """Balanced nested brackets with k types. Vocab = 2k symbols (k open, k close)."""

    def __init__(self, k: int = 3, max_depth: int = 12, p_open: float = 0.45, seed: int = 0):
        self.k = k
        self.max_depth = max_depth
        self.p_open = p_open
        self.vocab_size = 2 * k
        self.rng = random.Random(seed)

    def _emit(self, n: int):
        out, stack = [], []
        while len(out) < n:
            can_open = len(stack) < self.max_depth
            can_close = len(stack) > 0
            if can_open and (not can_close or self.rng.random() < self.p_open):
                t = self.rng.randrange(self.k)
                stack.append(t)
                out.append(t)                 # open bracket type t  -> id t
            elif can_close:
                t = stack.pop()
                out.append(self.k + t)        # close bracket type t -> id k+t
            else:
                t = self.rng.randrange(self.k)
                stack.append(t)
                out.append(t)
        return out[:n]

    def batch(self, batch_size: int, seq_len: int, device="cpu"):
        seqs = [self._emit(seq_len + 1) for _ in range(batch_size)]
        data = torch.tensor(seqs, dtype=torch.long, device=device)
        return data[:, :-1].contiguous(), data[:, 1:].contiguous()


class CharText:
    """Char-level LM over a local UTF-8 text file."""

    def __init__(self, path: str, device="cpu"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long, device=device)

    def batch(self, batch_size: int, seq_len: int, device="cpu"):
        ix = torch.randint(0, self.data.numel() - seq_len - 1, (batch_size,))
        x = torch.stack([self.data[i:i + seq_len] for i in ix]).to(device)
        y = torch.stack([self.data[i + 1:i + 1 + seq_len] for i in ix]).to(device)
        return x, y


def make_dataset(name: str, **kw):
    if name == "dyck":
        return DyckK(k=kw.get("k", 3), max_depth=kw.get("max_depth", 12), seed=kw.get("seed", 0))
    if name == "char":
        return CharText(kw["path"], device=kw.get("device", "cpu"))
    raise ValueError(f"unknown dataset {name!r}")
