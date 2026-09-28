"""Sanity checks for the tree generator. Run: python tests/test_trees.py"""
from curvlearn.trees import balanced_tree


def test_node_counts():
    for b, h, N in [(2, 4, 31), (2, 6, 127), (3, 2, 13), (2, 8, 511)]:
        n, D = balanced_tree(b, h)
        assert n == N, (b, h, n, N)
        assert tuple(D.shape) == (N, N)


def test_metric_properties():
    N, D = balanced_tree(2, 5)
    assert (D == D.T).all()                     # symmetric
    assert (D.diagonal() == 0).all()            # zero diagonal
    assert D.max().item() == 2 * 5              # tree diameter of complete binary tree = 2h
    # root (0) to any node = its depth; deepest leaf is at depth h
    assert D[0].max().item() == 5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
