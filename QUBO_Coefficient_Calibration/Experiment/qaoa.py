"""Shared QAOA execution helpers used by experiment modules.

This module contains solver mechanics only. Manuscript study definitions live in
explicit sibling modules (penalty, budget, spectral, optional, etc.).
"""
from __future__ import annotations

import math
import numpy as np

from QUBO_Coefficient_Calibration.Problem.cardinality import CardinalitySelectionQUBO
from QUBO_Coefficient_Calibration.Experiment.analysis import (
    enumerate_landscape,
    exact_local_threshold,
    exact_global_threshold,
    exact_band_threshold,
    one_bit_lipschitz_certificate,
    coefficient_range_certificate,
    global_certificate,
)

def _probability_metrics(problem, probs):
    land = enumerate_landscape(problem, optional_scale=0.0)
    probs = np.asarray(probs, dtype=float)
    feas = land["feasible"]
    p_feas = float(np.sum(probs[feas]))
    best_primary = float(np.min(land["primary"][feas]))
    optmask = feas & np.isclose(land["primary"], best_primary, atol=1e-12, rtol=0.0)
    p_opt = float(np.sum(probs[optmask]))
    if p_feas > 0.0:
        conditional_primary = float(np.sum(probs[feas] * land["primary"][feas]) / p_feas)
    else:
        conditional_primary = math.nan
    denom = max(abs(best_primary), 1.0)
    rel_gap = float((conditional_primary - best_primary) / denom) if np.isfinite(conditional_primary) else math.nan
    return {
        "p_feasible": p_feas,
        "p_primary_optimum": p_opt,
        "p_best_feasible": p_opt,  # old plotting schema
        "p_feasible_100_shots": float(1.0 - (1.0 - p_feas) ** 100),
        "p_primary_optimum_100_shots": float(1.0 - (1.0 - p_opt) ** 100),
        "uniform_p_feasible": float(problem.feasible_fraction),
        "conditional_expected_primary": conditional_primary,
        "exact_primary_optimum": best_primary,
        "relative_primary_gap": rel_gap,
    }


def _make_solver(qubo, *, engine, p, qseed, optimizer, device_name):
    if str(engine).lower().strip() != "pennylane":
        raise ValueError("only the 'pennylane' QAOA engine is included in this trimmed repository")
    from QUBO_Coefficient_Calibration.Solver.pennylane_qaoa import QAOASolver
    return QAOASolver(
        qubo, p=int(p), device_name=device_name, shots=None, seed=qseed,
        mixer_kind="transverse", optimizer_kind=optimizer, normalize_cost=True,
    )


def _solve_multistart(problem, *, penalty, optional_scale, p, restarts, maxiter,
                      stepsize, optimizer, device_name, seed_base, engine, seed_stride):
    restart_results = []
    for r in range(int(restarts)):
        qseed = int(seed_base + 10007 * int(problem.seed) + int(seed_stride) * r)
        qubo = CardinalitySelectionQUBO(problem, penalty=float(penalty), optional_scale=float(optional_scale))
        solver = _make_solver(
            qubo, engine=engine, p=p, qseed=qseed, optimizer=optimizer, device_name=device_name,
        )
        result = solver.solve(
            maxiter=int(maxiter), stepsize=float(stepsize), tol=1e-7,
            top_k=min(1 << problem.n, 128), return_probabilities=True,
        )
        metrics = _probability_metrics(problem, result["final_probabilities"])
        history = np.asarray(result["history"], dtype=float)
        restart_results.append((
            float(result["expval_energy_optimization_scale"]), result, metrics, qseed, r, history
        ))

    # Multi-start selection uses only the variational objective, never oracle labels.
    restart_results.sort(key=lambda t: t[0])
    return restart_results[0]


def _solve_multistart_checkpoints(
    problem,
    *,
    penalty,
    optional_scale,
    p,
    restarts,
    checkpoints,
    stepsize,
    optimizer,
    device_name,
    seed_base,
    engine,
    seed_stride,
):
    """Run each restart once and select the best restart independently at each checkpoint.

    This is the paired-budget protocol used by ``budget_heatmap.pdf``.  Every
    restart has one initialization seed and one uninterrupted optimizer trajectory
    up to ``max(checkpoints)``.  At work budget B, multi-start selection uses the
    variational expectation value *at checkpoint B*, never a separately rerun
    optimization and never oracle feasibility/quality labels.
    """
    requested = sorted({int(v) for v in checkpoints if int(v) > 0})
    if not requested:
        return {}
    maxiter = int(max(requested))
    by_checkpoint = {b: [] for b in requested}

    for r in range(int(restarts)):
        qseed = int(seed_base + 10007 * int(problem.seed) + int(seed_stride) * r)
        qubo = CardinalitySelectionQUBO(
            problem, penalty=float(penalty), optional_scale=float(optional_scale)
        )
        solver = _make_solver(
            qubo, engine=engine, p=p, qseed=qseed, optimizer=optimizer, device_name=device_name,
        )
        result = solver.solve(
            maxiter=maxiter,
            stepsize=float(stepsize),
            tol=1e-7,
            top_k=min(1 << problem.n, 128),
            return_probabilities=False,
            checkpoint_iterations=requested,
        )

        checkpoints_result = result.get("checkpoint_results", {})
        for budget in requested:
            snap = checkpoints_result.get(int(budget))
            if snap is None:
                raise RuntimeError(
                    f"QAOA checkpoint {budget} was requested but not returned by {engine}."
                )
            metrics = _probability_metrics(problem, snap["final_probabilities"])
            by_checkpoint[budget].append((
                float(snap["expval_energy_optimization_scale"]),
                snap,
                metrics,
                qseed,
                r,
            ))

    selected = {}
    for budget in requested:
        candidates = by_checkpoint[budget]
        candidates.sort(key=lambda t: t[0])
        selected[budget] = candidates[0]
    return selected


def _penalty_marker_fields(problem):
    pl = float(exact_local_threshold(problem, optional_scale=0.0))
    pg = float(exact_global_threshold(problem, optional_scale=0.0))
    pb = float(exact_band_threshold(problem, optional_scale=0.0))
    return {
        "P_local_tight_boundary": pl,
        "P_local_surrogate": float(one_bit_lipschitz_certificate(problem, 0.0)),
        "P_global_tight_boundary": pg,
        "P_global_surrogate": float(global_certificate(problem, 0.0)["tight"]),
        "P_band_tight_boundary": pb,
        "P_band_surrogate": float(coefficient_range_certificate(problem, 0.0)),
    }
