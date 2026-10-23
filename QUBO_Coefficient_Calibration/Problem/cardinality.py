"""Fixed-cardinality binary quadratic case study for coefficient calibration.

The model is deliberately small and theory-facing:

    H_pri(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j
    H_opt(x; lambda) = lambda * H_opt_template(x)
    C(x) = (sum_i x_i - k)^2
    H(x) = H_pri(x) + H_opt(x; lambda) + P C(x)

There are no slack/auxiliary variables.  For k near n/2 the feasible set is
large (binomial(n,k) states), which makes it much friendlier to transverse-field
QAOA than the routing encoding while preserving the paper's abstract penalty and
optional-ordering questions.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from itertools import combinations
from math import comb
from typing import Dict, Tuple, Any, Optional

import numpy as np

Pair = Tuple[int, int]


def _clean_pair(i: int, j: int) -> Pair:
    if i == j:
        raise ValueError("Quadratic pair must use distinct indices")
    return (i, j) if i < j else (j, i)


@dataclass(frozen=True)
class CardinalitySelectionProblem:
    n: int
    k: int
    primary_linear: np.ndarray
    primary_quadratic: Dict[Pair, int]
    optional_linear: np.ndarray
    optional_quadratic: Dict[Pair, int]
    seed: int = 0

    def __post_init__(self):
        n = int(self.n)
        k = int(self.k)
        if n <= 1:
            raise ValueError("n must be >= 2")
        if not (0 < k < n):
            raise ValueError("k must satisfy 0 < k < n")
        if np.asarray(self.primary_linear).shape != (n,):
            raise ValueError("primary_linear must have shape (n,)")
        if np.asarray(self.optional_linear).shape != (n,):
            raise ValueError("optional_linear must have shape (n,)")
        for mapping in (self.primary_quadratic, self.optional_quadratic):
            for (i, j) in mapping:
                if not (0 <= i < j < n):
                    raise ValueError(f"Invalid quadratic index {(i, j)} for n={n}")

    @property
    def feasible_count(self) -> int:
        return int(comb(int(self.n), int(self.k)))

    @property
    def feasible_fraction(self) -> float:
        return float(self.feasible_count / (2 ** int(self.n)))

    def primary_value(self, bits) -> float:
        x = np.asarray(bits, dtype=float)
        val = float(np.dot(np.asarray(self.primary_linear, dtype=float), x))
        for (i, j), c in self.primary_quadratic.items():
            val += float(c) * float(x[i]) * float(x[j])
        return float(val)

    def optional_template_value(self, bits) -> float:
        x = np.asarray(bits, dtype=float)
        val = float(np.dot(np.asarray(self.optional_linear, dtype=float), x))
        for (i, j), c in self.optional_quadratic.items():
            val += float(c) * float(x[i]) * float(x[j])
        return float(val)

    def violation(self, bits) -> float:
        x = np.asarray(bits, dtype=int)
        return float((int(np.sum(x)) - int(self.k)) ** 2)

    def feasible(self, bits) -> bool:
        return bool(int(np.sum(np.asarray(bits, dtype=int))) == int(self.k))


class CardinalitySelectionQUBO:
    """QUBO adapter compatible with the project's existing ``QAOASolver``.

    ``x_index`` intentionally uses the historical 3-tuple key shape so the old
    VRP decoder remains harmless.  There are no y/slack variables.
    """

    def __init__(self, problem: CardinalitySelectionProblem, penalty: float, optional_scale: float = 0.0):
        self.problem = problem
        self.penalty = float(penalty)
        self.optional_scale = float(optional_scale)
        if self.penalty < 0 or not np.isfinite(self.penalty):
            raise ValueError("penalty must be finite and >= 0")
        if not np.isfinite(self.optional_scale):
            raise ValueError("optional_scale must be finite")

        self.x_index = {(i, 0, 0): i for i in range(problem.n)}
        self.y_index = {}

        self.offset = 0.0
        self.linear: Dict[int, float] = defaultdict(float)
        self.quadratic: Dict[Pair, float] = defaultdict(float)

        # Existing solver names: ``rew`` is the fixed primary objective, while
        # ``base`` is primary + optional and excludes penalties.
        self.offset_rew = 0.0
        self.linear_rew: Dict[int, float] = defaultdict(float)
        self.quadratic_rew: Dict[Pair, float] = defaultdict(float)
        self.offset_base = 0.0
        self.linear_base: Dict[int, float] = defaultdict(float)
        self.quadratic_base: Dict[Pair, float] = defaultdict(float)
        self.build()

    def _add_linear(self, mapping, i: int, c: float):
        if c:
            mapping[int(i)] += float(c)

    def _add_quadratic(self, mapping, i: int, j: int, c: float):
        if c:
            mapping[_clean_pair(int(i), int(j))] += float(c)

    def build(self):
        self.offset = 0.0
        self.linear = defaultdict(float)
        self.quadratic = defaultdict(float)
        self.offset_rew = 0.0
        self.linear_rew = defaultdict(float)
        self.quadratic_rew = defaultdict(float)
        self.offset_base = 0.0
        self.linear_base = defaultdict(float)
        self.quadratic_base = defaultdict(float)

        p = self.problem
        lam = self.optional_scale

        # Primary objective.
        for i, c in enumerate(np.asarray(p.primary_linear, dtype=float)):
            self._add_linear(self.linear_rew, i, c)
            self._add_linear(self.linear_base, i, c)
            self._add_linear(self.linear, i, c)
        for (i, j), c in p.primary_quadratic.items():
            self._add_quadratic(self.quadratic_rew, i, j, c)
            self._add_quadratic(self.quadratic_base, i, j, c)
            self._add_quadratic(self.quadratic, i, j, c)

        # Optional landscape shaping term.
        for i, c in enumerate(np.asarray(p.optional_linear, dtype=float)):
            cc = lam * float(c)
            self._add_linear(self.linear_base, i, cc)
            self._add_linear(self.linear, i, cc)
        for (i, j), c in p.optional_quadratic.items():
            cc = lam * float(c)
            self._add_quadratic(self.quadratic_base, i, j, cc)
            self._add_quadratic(self.quadratic, i, j, cc)

        # P (sum_i x_i - k)^2.  Since x_i^2=x_i:
        # P[k^2 + (1-2k)sum_i x_i + 2sum_{i<j}x_i x_j].
        P = self.penalty
        k = int(p.k)
        self.offset += P * (k ** 2)
        for i in range(p.n):
            self._add_linear(self.linear, i, P * (1 - 2 * k))
        for i, j in combinations(range(p.n), 2):
            self._add_quadratic(self.quadratic, i, j, 2.0 * P)
        return self

    @staticmethod
    def _poly_value(offset: float, linear: Dict[int, float], quadratic: Dict[Pair, float], bits) -> float:
        x = np.asarray(bits, dtype=float)
        val = float(offset)
        for i, c in linear.items():
            val += float(c) * float(x[i])
        for (i, j), c in quadratic.items():
            val += float(c) * float(x[i]) * float(x[j])
        return float(val)

    def value(self, bits) -> float:
        return self._poly_value(self.offset, self.linear, self.quadratic, bits)

    def value_rew(self, bits) -> float:
        return self._poly_value(self.offset_rew, self.linear_rew, self.quadratic_rew, bits)

    def value_base(self, bits) -> float:
        return self._poly_value(self.offset_base, self.linear_base, self.quadratic_base, bits)

    def validate_bits(self, bits, **_kwargs) -> Dict[str, Any]:
        x = np.asarray(bits, dtype=int)
        length_ok = bool(x.size == self.problem.n)
        binary_ok = bool(length_ok and np.all((x == 0) | (x == 1)))
        weight = int(np.sum(x)) if length_ok else None
        feasible = bool(binary_ok and weight == self.problem.k)
        return {
            "feasible": feasible,
            "assignment_feasible": feasible,
            "length_ok": length_ok,
            "binary_ok": binary_ok,
            "cardinality": weight,
            "target_cardinality": int(self.problem.k),
            "violation": (float((weight - self.problem.k) ** 2) if weight is not None else float("inf")),
        }

    def decode_bits(self, bits) -> Dict[str, Any]:
        x = np.asarray(bits, dtype=int)
        return {
            "selected": [int(i) for i in np.flatnonzero(x)],
            "x": {int(i): int(x[i]) for i in range(self.problem.n)},
            "cardinality": int(np.sum(x)),
        }

    def to_ising(self) -> Dict[str, Any]:
        c0 = float(self.offset)
        h = defaultdict(float)
        J = defaultdict(float)
        for i, a in self.linear.items():
            c0 += 0.5 * float(a)
            h[i] += -0.5 * float(a)
        for (i, j), b in self.quadratic.items():
            b = float(b)
            c0 += 0.25 * b
            h[i] += -0.25 * b
            h[j] += -0.25 * b
            J[(i, j)] += 0.25 * b
        return {
            "offset": float(c0),
            "linear": {int(i): float(v) for i, v in h.items()},
            "quadratic": {f"{i},{j}": float(v) for (i, j), v in J.items()},
        }


def build_random_cardinality_problem(
    *,
    seed: int,
    n: int = 8,
    k: Optional[int] = None,
    primary_density: float = 0.45,
    optional_density: float = 0.20,
    require_penalty_nontrivial: bool = True,
    require_finite_ordering_boundary: bool = True,
    max_attempts: int = 500,
) -> CardinalitySelectionProblem:
    """Generate a reproducible, nontrivial small case-study instance.

    The structural filters are intentional and should be reported in the paper:
    * ``require_penalty_nontrivial`` rejects instances whose unconstrained primary
      optimum already has the target cardinality, because they cannot demonstrate
      why a penalty is needed.
    * ``require_finite_ordering_boundary`` rejects optional templates that never
      oppose any strict primary comparison on the feasible set, because such an
      instance has no finite ordering-failure scale to validate.

    These filters do not inspect QAOA performance.
    """
    n = int(n)
    k = int(n // 2 if k is None else k)
    if not (0 < k < n):
        raise ValueError("Require 0 < k < n")
    rng = np.random.default_rng(int(seed))

    from QUBO_Coefficient_Calibration.Experiment.analysis import enumerate_landscape, exact_ordering_threshold

    for _attempt in range(int(max_attempts)):
        a = rng.integers(-4, 5, size=n, dtype=int)
        if not np.any(a):
            a[int(rng.integers(0, n))] = 1

        b: Dict[Pair, int] = {}
        for i, j in combinations(range(n), 2):
            if rng.random() < float(primary_density):
                c = 0
                while c == 0:
                    c = int(rng.integers(-3, 4))
                b[(i, j)] = c

        u = rng.integers(-2, 3, size=n, dtype=int)
        v: Dict[Pair, int] = {}
        for i, j in combinations(range(n), 2):
            if rng.random() < float(optional_density):
                c = int(rng.choice([-1, 1]))
                v[(i, j)] = c
        if not np.any(u) and not v:
            u[int(rng.integers(0, n))] = 1

        problem = CardinalitySelectionProblem(
            n=n,
            k=k,
            primary_linear=a,
            primary_quadratic=b,
            optional_linear=u,
            optional_quadratic=v,
            seed=int(seed),
        )
        land = enumerate_landscape(problem, optional_scale=0.0)
        feasible = land["feasible"]
        pri = land["primary"]

        if require_penalty_nontrivial:
            global_min_mask = np.isclose(pri, float(np.min(pri)), atol=1e-12, rtol=0.0)
            if bool(np.any(feasible & global_min_mask)):
                continue

        if require_finite_ordering_boundary:
            threshold = exact_ordering_threshold(problem)
            if not np.isfinite(threshold) or threshold <= 0.0:
                continue

        return problem

    raise RuntimeError(
        f"Could not generate a nontrivial cardinality instance after {max_attempts} attempts "
        f"(seed={seed}, n={n}, k={k})."
    )
