"""Exact theorem-facing diagnostics for the fixed-cardinality case study."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Any
import math
import numpy as np


def all_bitstrings(n: int) -> np.ndarray:
    idx = np.arange(1 << int(n), dtype=np.uint64)[:, None]
    shifts = np.arange(int(n) - 1, -1, -1, dtype=np.uint64)[None, :]
    return ((idx >> shifts) & 1).astype(np.int8)


def _poly_batch(bits: np.ndarray, linear, quadratic) -> np.ndarray:
    vals = bits @ np.asarray(linear, dtype=float)
    vals = np.asarray(vals, dtype=float)
    for (i, j), c in quadratic.items():
        vals += float(c) * bits[:, i] * bits[:, j]
    return vals


def enumerate_landscape(problem, optional_scale: float = 0.0) -> Dict[str, Any]:
    bits = all_bitstrings(problem.n)
    primary = _poly_batch(bits, problem.primary_linear, problem.primary_quadratic)
    optional_template = _poly_batch(bits, problem.optional_linear, problem.optional_quadratic)
    base = primary + float(optional_scale) * optional_template
    weight = bits.sum(axis=1).astype(int)
    violation = (weight - int(problem.k)) ** 2
    feasible = violation == 0
    return {
        "bits": bits,
        "primary": primary,
        "optional_template": optional_template,
        "base": base,
        "weight": weight,
        "violation": violation.astype(float),
        "feasible": feasible,
    }


def exact_local_threshold(problem, optional_scale: float = 0.0) -> float:
    land = enumerate_landscape(problem, optional_scale)
    bits, base, feasible = land["bits"], land["base"], land["feasible"]
    # Dictionary from bit tuple to row for O(n 2^n) lookup.
    row = {tuple(map(int, b)): i for i, b in enumerate(bits)}
    worst = 0.0
    for i in np.flatnonzero(feasible):
        x = bits[i].copy()
        for q in range(problem.n):
            y = x.copy(); y[q] ^= 1
            j = row[tuple(map(int, y))]
            # A one-bit move from a k-cardinality state has C(y)=1.
            worst = max(worst, float(base[i] - base[j]), 0.0)
    return float(worst)


def exact_global_threshold(problem, optional_scale: float = 0.0) -> float:
    land = enumerate_landscape(problem, optional_scale)
    base, vio, feas = land["base"], land["violation"], land["feasible"]
    best_feasible = float(np.min(base[feas]))
    ratios = (best_feasible - base[~feas]) / vio[~feas]
    return float(max(0.0, float(np.max(ratios, initial=0.0))))


def exact_band_threshold(problem, optional_scale: float = 0.0) -> float:
    land = enumerate_landscape(problem, optional_scale)
    base, vio, feas = land["base"], land["violation"], land["feasible"]
    worst_feasible = float(np.max(base[feas]))
    ratios = (worst_feasible - base[~feas]) / vio[~feas]
    return float(max(0.0, float(np.max(ratios, initial=0.0))))


def one_bit_lipschitz_certificate(problem, optional_scale: float = 0.0) -> float:
    linear = np.asarray(problem.primary_linear, dtype=float) + float(optional_scale) * np.asarray(problem.optional_linear, dtype=float)
    quad = {k: float(v) for k, v in problem.primary_quadratic.items()}
    for pair, c in problem.optional_quadratic.items():
        quad[pair] = quad.get(pair, 0.0) + float(optional_scale) * float(c)
    incident = np.abs(linear).astype(float)
    for (i, j), c in quad.items():
        incident[i] += abs(float(c))
        incident[j] += abs(float(c))
    return float(np.max(incident, initial=0.0))


def coefficient_range_certificate(problem, optional_scale: float = 0.0) -> float:
    linear = np.asarray(problem.primary_linear, dtype=float) + float(optional_scale) * np.asarray(problem.optional_linear, dtype=float)
    total = float(np.sum(np.abs(linear)))
    pairs = set(problem.primary_quadratic) | set(problem.optional_quadratic)
    for pair in pairs:
        c = float(problem.primary_quadratic.get(pair, 0.0)) + float(optional_scale) * float(problem.optional_quadratic.get(pair, 0.0))
        total += abs(c)
    return float(total)


def coefficientwise_lower_bound(problem, optional_scale: float = 0.0) -> float:
    linear = np.asarray(problem.primary_linear, dtype=float) + float(optional_scale) * np.asarray(problem.optional_linear, dtype=float)
    lower = float(np.sum(np.minimum(linear, 0.0)))
    pairs = set(problem.primary_quadratic) | set(problem.optional_quadratic)
    for pair in pairs:
        c = float(problem.primary_quadratic.get(pair, 0.0)) + float(optional_scale) * float(problem.optional_quadratic.get(pair, 0.0))
        lower += min(c, 0.0)
    return float(lower)


def greedy_feasible_incumbent(problem, optional_scale: float = 0.0) -> float:
    """Simple deterministic feasible incumbent used only for the value certificate.

    Start with the k variables having smallest effective linear coefficient, then
    apply improving one-for-one swaps until locally optimal on the feasible set.
    """
    n, k = int(problem.n), int(problem.k)
    linear = np.asarray(problem.primary_linear, dtype=float) + float(optional_scale) * np.asarray(problem.optional_linear, dtype=float)
    selected = set(np.argsort(linear, kind="stable")[:k].tolist())

    def value(sel):
        x = np.zeros(n, dtype=int); x[list(sel)] = 1
        return float(problem.primary_value(x) + float(optional_scale) * problem.optional_template_value(x))

    current = value(selected)
    improved = True
    while improved:
        improved = False
        best_val = current
        best_sel = None
        outside = [i for i in range(n) if i not in selected]
        for i in sorted(selected):
            for j in outside:
                cand = (selected - {i}) | {j}
                v = value(cand)
                if v < best_val - 1e-12:
                    best_val, best_sel = v, cand
        if best_sel is not None:
            selected, current, improved = best_sel, best_val, True
    return float(current)


def global_certificate(problem, optional_scale: float = 0.0) -> Dict[str, float]:
    B = coefficient_range_certificate(problem, optional_scale)
    delta = one_bit_lipschitz_certificate(problem, optional_scale)
    lower = coefficientwise_lower_bound(problem, optional_scale)
    incumbent = greedy_feasible_incumbent(problem, optional_scale)
    value_cert = max(0.0, incumbent - lower)
    # Cardinality repair: a state at distance d=|sum x-k| can be repaired in d
    # flips, while C=d^2.  d*Delta / d^2 <= Delta for d>=1.
    repair_cert = delta
    tight = min(B, value_cert, repair_cert)
    return {
        "range": float(B),
        "one_bit": float(delta),
        "lower_bound": float(lower),
        "incumbent": float(incumbent),
        "value": float(value_cert),
        "repair": float(repair_cert),
        "tight": float(tight),
    }


def exact_ordering_threshold(problem) -> float:
    """Sharp first strict-order failure, using bounded-memory vectorized pairs.

    Only pairs with primary(i) < primary(j) and optional(j) < optional(i)
    threaten ordering.  Process rows in small blocks to avoid a giant NxN
    array at n=14..16.  The calculation is independent of the separately
    implemented primary-level grouping in optional.exact_thresholds.
    """
    land = enumerate_landscape(problem, optional_scale=0.0)
    pri = np.asarray(land["primary"][land["feasible"]], dtype=float)
    opt = np.asarray(land["optional_template"][land["feasible"]], dtype=float)
    best = math.inf
    for first in range(0, len(pri), 96):
        p = pri[first:first + 96]
        o = opt[first:first + 96]
        dpri = pri[None, :] - p[:, None]
        dopt = opt[None, :] - o[:, None]
        threatening = (dpri > 1e-12) & (dopt < -1e-12)
        if threatening.any():
            best = min(best, float(np.min(dpri[threatening] / (-dopt[threatening]))))
    return float(best)


def exact_primary_gap(problem) -> float:
    land = enumerate_landscape(problem, optional_scale=0.0)
    vals = np.unique(np.asarray(land["primary"][land["feasible"]], dtype=float))
    if len(vals) < 2:
        return math.inf
    return float(np.min(np.diff(np.sort(vals))))


def exact_optional_variation(problem) -> float:
    land = enumerate_landscape(problem, optional_scale=0.0)
    vals = land["optional_template"][land["feasible"]]
    return float(np.max(vals) - np.min(vals))


def optional_variation_certificate(problem) -> float:
    return float(
        np.sum(np.abs(np.asarray(problem.optional_linear, dtype=float)))
        + sum(abs(float(c)) for c in problem.optional_quadratic.values())
    )


def ordering_certificates(problem) -> Dict[str, float]:
    dpri = exact_primary_gap(problem)
    dopt_exact = exact_optional_variation(problem)
    dopt_cert = optional_variation_certificate(problem)
    exact_sufficient = math.inf if dopt_exact <= 0 else dpri / dopt_exact
    # Integer primary coefficients imply every nonzero primary gap is >=1.
    practical = math.inf if dopt_cert <= 0 else 1.0 / dopt_cert
    return {
        "oracle": exact_ordering_threshold(problem),
        "primary_gap_exact": float(dpri),
        "primary_gap_integer_lower": 1.0,
        "optional_variation_exact": float(dopt_exact),
        "optional_variation_certificate": float(dopt_cert),
        "exact_sufficient": float(exact_sufficient),
        "practical": float(practical),
    }


def ordering_preserved(problem, optional_scale: float) -> bool:
    land = enumerate_landscape(problem, optional_scale=0.0)
    feas = land["feasible"]
    pri = land["primary"][feas]
    base = pri + float(optional_scale) * land["optional_template"][feas]
    m = len(pri)
    for i in range(m):
        for j in range(m):
            if pri[i] < pri[j] - 1e-12 and not (base[i] < base[j] - 1e-12):
                return False
    return True


def property_holds(problem, property_name: str, penalty: float, optional_scale: float = 0.0, strict_tol: float = 1e-10) -> bool:
    land = enumerate_landscape(problem, optional_scale)
    base, vio, feas = land["base"], land["violation"], land["feasible"]
    H = base + float(penalty) * vio
    name = str(property_name).lower()
    if name == "global":
        return bool(np.min(H[~feas]) > np.min(H[feas]) + strict_tol)
    if name == "band":
        return bool(np.min(H[~feas]) > np.max(H[feas]) + strict_tol)
    if name == "local":
        bits = land["bits"]
        row = {tuple(map(int, b)): i for i, b in enumerate(bits)}
        for i in np.flatnonzero(feas):
            x = bits[i]
            for q in range(problem.n):
                y = x.copy(); y[q] ^= 1
                j = row[tuple(map(int, y))]
                if not (H[j] > H[i] + strict_tol):
                    return False
        return True
    raise ValueError("property_name must be local, global, or band")


def local_diagnostics(problem, penalty: float, optional_scale: float = 0.0, atol: float = 1e-10) -> Dict[str, Any]:
    """Exact local-boundary diagnostics matching the manuscript plot semantics."""
    land = enumerate_landscape(problem, optional_scale)
    bits, base, vio, feas = land["bits"], land["base"], land["violation"], land["feasible"]
    energy = base + float(penalty) * vio
    row = {tuple(map(int, b)): i for i, b in enumerate(bits)}
    gaps = []
    for i in np.flatnonzero(feas):
        x = bits[i]
        for q in range(problem.n):
            y = x.copy(); y[q] ^= 1
            j = row[tuple(map(int, y))]
            if not feas[j]:
                gaps.append(float(energy[j] - energy[i]))
    if not gaps:
        return {"g_l": math.nan, "Gamma_l": math.nan}
    arr = np.asarray(gaps, dtype=float)
    return {
        "g_l": float(np.min(arr)),
        "Gamma_l": float(np.mean(arr < -float(atol))),
    }


def global_diagnostics(problem, penalty: float, optional_scale: float = 0.0, atol: float = 1e-10) -> Dict[str, Any]:
    """Exact global-minimizer feasibility gap for one penalty."""
    land = enumerate_landscape(problem, optional_scale)
    energy = land["base"] + float(penalty) * land["violation"]
    feas = land["feasible"]
    ef = float(np.min(energy[feas]))
    ei = float(np.min(energy[~feas]))
    gap = float(ei - ef)
    return {"g_g": gap, "all_global_minimizers_feasible": bool(gap > float(atol))}


def band_diagnostics(problem, penalty: float, optional_scale: float = 0.0, atol: float = 1e-10) -> Dict[str, Any]:
    """Exact complete feasible/infeasible band gap and violating-pair fraction."""
    land = enumerate_landscape(problem, optional_scale)
    energy = land["base"] + float(penalty) * land["violation"]
    feas = land["feasible"]
    f = np.asarray(energy[feas], dtype=float)
    inf = np.asarray(energy[~feas], dtype=float)
    gap = float(np.min(inf) - np.max(f))
    inf_sorted = np.sort(inf)
    violating = int(np.searchsorted(inf_sorted, f, side="right").sum())
    gamma = violating / float(f.size * inf.size)
    return {"g_u": gap, "Gamma_u": float(gamma), "strict_band_separation": bool(gap > float(atol))}


def exact_base_range(problem, optional_scale: float = 0.0) -> float:
    land = enumerate_landscape(problem, optional_scale)
    return float(np.max(land["base"]) - np.min(land["base"]))


def exact_infeasible_advantage(problem, optional_scale: float = 0.0) -> float:
    """G_F = max_I [H_F^* - H_base(I)]_+ before dividing by violation."""
    land = enumerate_landscape(problem, optional_scale)
    feas = land["feasible"]
    best_f = float(np.min(land["base"][feas]))
    return float(max(0.0, best_f - float(np.min(land["base"][~feas]))))


def ordering_diagnostics(problem, optional_scale: float, atol: float = 1e-12) -> Dict[str, Any]:
    """Pairwise strict-order diagnostics using the same quantities as the old plots."""
    land = enumerate_landscape(problem, optional_scale=0.0)
    feas = land["feasible"]
    pri = np.asarray(land["primary"][feas], dtype=float)
    full = pri + float(optional_scale) * np.asarray(land["optional_template"][feas], dtype=float)
    g = math.inf
    violations = 0
    pairs = 0
    for i in range(len(pri)):
        for j in range(len(pri)):
            if pri[i] < pri[j] - float(atol):
                d = float(full[j] - full[i])
                g = min(g, d)
                pairs += 1
                violations += int(d <= float(atol))
    if pairs == 0:
        g = math.nan
        gamma = math.nan
    else:
        gamma = float(violations / pairs)
    try:
        from scipy.stats import kendalltau
        tau = float(kendalltau(pri, full).statistic) if np.unique(pri).size > 1 and np.unique(full).size > 1 else math.nan
    except Exception:
        tau = math.nan
    pearson = float(np.corrcoef(pri, full)[0, 1]) if np.std(pri) > 0 and np.std(full) > 0 else math.nan
    return {"g_ord": float(g), "Gamma_ord": gamma, "kendall_tau": tau, "pearson_r": pearson}
