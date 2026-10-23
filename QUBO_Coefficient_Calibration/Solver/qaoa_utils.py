"""Pure helpers shared by the PennyLane QAOA implementation and tests."""
from __future__ import annotations
from typing import List, Tuple


def unique_ring_edges(n: int) -> List[Tuple[int, int]]:
    """Return the manuscript's unique undirected nearest-neighbour ring edges.

    Indices are 0-based in code.  R_1 is empty; R_2 contains exactly one edge,
    so the physical pair is never double-counted.
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    if n <= 1:
        return []
    edges = {(min(i, (i + 1) % n), max(i, (i + 1) % n)) for i in range(n)}
    edges.discard((0, 0))
    return sorted(edges)
