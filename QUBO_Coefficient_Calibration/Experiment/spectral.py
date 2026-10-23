"""Exact spectral-gap experiment on the manuscript penalty grid."""
from __future__ import annotations

import numpy as np

from QUBO_Coefficient_Calibration.Experiment.analysis import exact_global_threshold, enumerate_landscape
from QUBO_Coefficient_Calibration.Experiment.qaoa import _penalty_marker_fields

SPECTRAL_P_OVER_GLOBAL = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]

def run_spectral_instance(problem, penalty_multipliers=SPECTRAL_P_OVER_GLOBAL):
    """Exact cardinality spectrum on the old Q1/Q2 penalty grid."""
    pg = float(exact_global_threshold(problem, optional_scale=0.0))
    if pg <= 0.0:
        return []
    markers = _penalty_marker_fields(problem)
    land = enumerate_landscape(problem, optional_scale=0.0)
    rows = []
    for multiplier in penalty_multipliers:
        penalty = float(multiplier) * pg
        energy = np.asarray(land["primary"] + penalty * land["violation"], dtype=float)
        uniq = np.unique(np.round(energy, decimals=12))
        low_gap = float(uniq[1] - uniq[0]) if uniq.size >= 2 else 0.0
        energy_range = float(np.max(energy) - np.min(energy))
        normalized = float(low_gap / energy_range) if energy_range > 0.0 else 0.0
        rows.append({
            "seed": int(problem.seed), "n": int(problem.n), "k": int(problem.k),
            "penalty": penalty, "penalty_multiplier": float(multiplier),
            "penalty_multiplier_reference": "tight",
            "penalty_multiplier_reference_value": pg,
            "low_gap": low_gap, "energy_range": energy_range,
            "normalized_gap": normalized,
            **markers,
        })
    return rows
