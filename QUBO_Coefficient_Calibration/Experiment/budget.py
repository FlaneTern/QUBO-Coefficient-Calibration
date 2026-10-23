"""Paired optimizer-budget QAOA experiment."""
from __future__ import annotations

from QUBO_Coefficient_Calibration.Experiment.analysis import exact_global_threshold
from QUBO_Coefficient_Calibration.Experiment.qaoa import _solve_multistart_checkpoints, _penalty_marker_fields

BUDGET_P_OVER_GLOBAL = [0.50, 0.75, 1.00, 1.25, 2.0, 4.0]
DEFAULT_WORK_BUDGETS = [20, 50, 100, 150, 250]

def run_qaoa_budget_instance(
    problem,
    *,
    budgets=DEFAULT_WORK_BUDGETS,
    penalty_multipliers=BUDGET_P_OVER_GLOBAL,
    p: int = 3,
    restarts: int = 8,
    stepsize: float = 0.05,
    optimizer: str = "adam",
    device_name: str = "default.qubit",
    seed_base: int = 24680,
    engine: str = "pennylane",
):
    """Paired QAOA work-budget study using checkpoints on shared trajectories.

    For each instance/penalty/restart, QAOA is optimized once to the largest
    requested budget.  Probability and energy are evaluated at the requested
    checkpoints.  Thus B=20,50,... compares the same restart trajectories rather
    than independently reinitializing QAOA for each budget.
    """
    pg = float(exact_global_threshold(problem, optional_scale=0.0))
    if pg <= 0.0:
        return []
    requested = sorted({int(v) for v in budgets if int(v) > 0})
    if not requested:
        return []

    markers = _penalty_marker_fields(problem)
    rows = []
    for multiplier in penalty_multipliers:
        penalty = float(multiplier) * pg
        selected = _solve_multistart_checkpoints(
            problem,
            penalty=penalty,
            optional_scale=0.0,
            p=p,
            restarts=restarts,
            checkpoints=requested,
            stepsize=stepsize,
            optimizer=optimizer,
            device_name=device_name,
            seed_base=seed_base,
            engine=engine,
            seed_stride=193,
        )
        for budget in requested:
            _, snap, metrics, qseed, chosen_restart = selected[budget]
            rows.append({
                "seed": int(problem.seed), "n": int(problem.n), "k": int(problem.k),
                "solver": "qaoa", "optimizer_iterations": int(budget), "work_budget": int(budget),
                "penalty": penalty, "penalty_multiplier": float(multiplier),
                "penalty_multiplier_reference": "tight",
                "penalty_multiplier_reference_value": pg,
                "qaoa_depth": int(p), "qaoa_restarts": int(restarts),
                "qaoa_engine": str(engine), "qaoa_device": str(device_name),
                "selected_restart": int(chosen_restart), "selected_qaoa_seed": int(qseed),
                "qaoa_iterations_completed": int(snap.get("iterations_completed", budget)),
                "qaoa_final_expval": float(snap["expval_energy_optimization_scale"]),
                "budget_protocol": "paired_checkpoints",
                "trajectory_maxiter": int(max(requested)),
                "restart_seed_base": int(seed_base),
                "restart_seed_stride": 193,
                **markers, **metrics,
            })
    return rows
