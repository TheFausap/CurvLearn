"""Complete b-ary trees and their all-pairs (graph) distance matrices.

The tree-distance embedding test embeds these nodes in the kappa-stereographic space and asks
whether geodesic distance can match tree distance with low distortion. Trees are the canonical
case where hyperbolic geometry should win: the number of nodes grows exponentially with depth,
matching hyperbolic volume growth, so a tree embeds in hyperbolic space with far lower
distortion than in Euclidean space of the same (low) dimension.
"""
from __future__ import annotations
import collections
import torch


def balanced_tree(b: int, h: int):
    """Complete b-ary tree of height h in array layout (root = 0, children of i are
    b*i+1 .. b*i+b). Returns (N, D) with D[i,j] = tree path length (float32 N x N)."""
    N = sum(b ** d for d in range(h + 1))
    adj = [[] for _ in range(N)]
    for i in range(N):
        for c in range(1, b + 1):
            j = b * i + c
            if j < N:
                adj[i].append(j)
                adj[j].append(i)
    D = torch.zeros(N, N, dtype=torch.float32)
    for s in range(N):
        dist = [-1] * N
        dist[s] = 0
        q = collections.deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if dist[w] < 0:
                    dist[w] = dist[u] + 1
                    q.append(w)
        D[s] = torch.tensor(dist, dtype=torch.float32)
    return N, D
