"""Penalty-multiplier and penalty-Nx QAOA experiments."""
from __future__ import annotations

import math
import numpy as np

from QUBO_Coefficient_Calibration.Experiment.analysis import (
    enumerate_landscape, exact_local_threshold, exact_global_threshold,
    exact_band_threshold, one_bit_lipschitz_certificate,
    coefficient_range_certificate, global_certificate,
)
from QUBO_Coefficient_Calibration.Experiment.qaoa import _solve_multistart, _penalty_marker_fields

QAOA_P_OVER_GLOBAL = [0.50, 0.75, 1.00, 1.25, 1.50, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]

def run_qaoa_penalty_instance(
    problem,
    *,
    penalty_multipliers=QAOA_P_OVER_GLOBAL,
    p: int = 3,
    restarts: int = 8,
    maxiter: int = 200,
    stepsize: float = 0.05,
    optimizer: str = "adam",
    device_name: str = "default.qubit",
    seed_base: int = 12345,
    engine: str = "pennylane",
):
    engine = str(engine).lower().strip()
    pg = float(exact_global_threshold(problem, optional_scale=0.0))
    if pg <= 0.0:
        return []
    markers = _penalty_marker_fields(problem)
    rows = []
    for multiplier in penalty_multipliers:
        penalty = float(multiplier) * pg
        _, result, metrics, qseed, chosen_restart, history = _solve_multistart(
            problem, penalty=penalty, optional_scale=0.0, p=p, restarts=restarts,
            maxiter=maxiter, stepsize=stepsize, optimizer=optimizer,
            device_name=device_name, seed_base=seed_base, engine=engine, seed_stride=97,
        )
        rows.append({
            "seed": int(problem.seed), "n": int(problem.n), "k": int(problem.k),
            "penalty": penalty,
            "penalty_multiplier": float(multiplier),
            "penalty_multiplier_reference": "tight",
            "penalty_multiplier_reference_value": pg,
            "qaoa_p": int(p), "qaoa_depth": int(p), "qaoa_restarts": int(restarts),
            "qaoa_engine": engine,
            "qaoa_device": str(device_name),
            "qaoa_optimizer": optimizer,
            "qaoa_stepsize": float(stepsize),
            "qaoa_maxiter": int(maxiter), "selected_restart": int(chosen_restart),
            "selected_qaoa_seed": int(qseed),
            "qaoa_iterations_completed": int(len(history)),
            "qaoa_initial_expval": float(history[0]) if len(history) else math.nan,
            "qaoa_final_expval": float(result["expval_energy_optimization_scale"]),
            "qaoa_best_expval_seen": float(np.min(history)) if len(history) else math.nan,
            "qaoa_energy_improvement": float(history[0] - np.min(history)) if len(history) else math.nan,
            "cost_scale_kappa": float(result["cost_scale_kappa"]),
            # Exact constrained enumeration is the penalty-independent classical reference.
            "milp_reference_has_incumbent": True,
            "milp_reference_feasible": True,
            "reference_label": "Exact constrained reference",
            **markers,
            **metrics,
        })
    return rows


def run_qaoa_scaling_instance(
    problem,
    *,
    p: int = 3,
    restarts: int = 4,
    maxiter: int = 150,
    stepsize: float = 0.05,
    optimizer: str = "adam",
    device_name: str = "default.qubit",
    seed_base: int = 314159,
    engine: str = "pennylane",
):
    pl = float(exact_local_threshold(problem, 0.0)); plc = float(one_bit_lipschitz_certificate(problem, 0.0))
    pg = float(exact_global_threshold(problem, 0.0)); pgc = float(global_certificate(problem, 0.0)["tight"])
    pb = float(exact_band_threshold(problem, 0.0)); pbc = float(coefficient_range_certificate(problem, 0.0))
    regimes = {
        "under": max(1e-9, 0.5 * pg),
        "local_tight": max(1e-9, 1.05 * pl),
        "local_surrogate": max(1e-9, 1.05 * plc),
        "global_tight": max(1e-9, 1.05 * pg),
        "global_surrogate": max(1e-9, 1.05 * pgc),
        "band_tight": max(1e-9, 1.05 * pb),
        "band_surrogate": max(1e-9, 1.05 * pbc),
    }
    reporting_penalty = max(1e-9, pg)
    land = enumerate_landscape(problem, 0.0)
    rows = []
    for ridx, (regime, penalty) in enumerate(regimes.items()):
        _, result, metrics, qseed, chosen_restart, history = _solve_multistart(
            problem, penalty=penalty, optional_scale=0.0, p=p, restarts=restarts,
            maxiter=maxiter, stepsize=stepsize, optimizer=optimizer,
            device_name=device_name, seed_base=seed_base + 7919 * ridx,
            engine=engine, seed_stride=211,
        )
        probs = np.asarray(result["final_probabilities"], dtype=float)
        report_energy = float(np.sum(probs * (land["primary"] + reporting_penalty * land["violation"])))
        rows.append({
            "seed": int(problem.seed), "N_x": int(problem.n), "n": int(problem.n), "k": int(problem.k),
            "penalty_regime": regime, "penalty": float(penalty),
            "reporting_penalty_reference": "exact-tight global-feasibility scale",
            "expected_H_primary_plus_reporting_penalty": report_energy,
            "qaoa_depth": int(p), "qaoa_restarts": int(restarts),
            "qaoa_engine": str(engine), "qaoa_device": str(device_name),
            "selected_restart": int(chosen_restart), "selected_qaoa_seed": int(qseed),
            "qaoa_iterations_completed": int(len(history)),
            **metrics,
        })
    return rows
