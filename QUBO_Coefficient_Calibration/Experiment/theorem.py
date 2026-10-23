"""Theorem-threshold and ordering experiments for fixed-cardinality QUBOs."""
from __future__ import annotations

from typing import Dict, Any, Sequence
import math
import numpy as np

from QUBO_Coefficient_Calibration.Experiment.analysis import (
    exact_local_threshold, exact_global_threshold, exact_band_threshold,
    one_bit_lipschitz_certificate, coefficient_range_certificate,
    global_certificate, ordering_certificates, ordering_preserved,
    property_holds, local_diagnostics, global_diagnostics, band_diagnostics,
    exact_base_range, exact_infeasible_advantage, ordering_diagnostics,
)

THEOREM_PENALTY_ORACLE_ETA_RATIOS = [
    -0.90, -0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50,
    1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
]
THEOREM_ORDER_ORACLE_ETA_RATIOS = [
    -1.0, -0.75, -0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50, 1.0, 3.0,
]

def _ratio_eta(value: float, oracle: float) -> float:
    if not np.isfinite(value) or not np.isfinite(oracle) or oracle <= 0.0:
        return math.nan
    return float(value / oracle - 1.0)


def theorem_summary(problem, optional_scale: float = 0.15) -> Dict[str, Any]:
    g = global_certificate(problem, optional_scale)
    o = ordering_certificates(problem)
    return {
        "seed": int(problem.seed),
        "n": int(problem.n),
        "k": int(problem.k),
        "feasible_count": int(problem.feasible_count),
        "state_count": int(2 ** problem.n),
        "feasible_fraction": float(problem.feasible_fraction),
        "optional_scale_for_penalty_theorems": float(optional_scale),
        "local_oracle": exact_local_threshold(problem, optional_scale),
        "local_certificate": one_bit_lipschitz_certificate(problem, optional_scale),
        "global_oracle": exact_global_threshold(problem, optional_scale),
        "global_certificate": g["tight"],
        "global_certificate_value": g["value"],
        "global_certificate_repair": g["repair"],
        "global_certificate_range": g["range"],
        "band_oracle": exact_band_threshold(problem, optional_scale),
        "band_certificate": coefficient_range_certificate(problem, optional_scale),
        "ordering_oracle": o["oracle"],
        "ordering_exact_sufficient": o["exact_sufficient"],
        "ordering_practical": o["practical"],
        "primary_gap_exact": o["primary_gap_exact"],
        "optional_variation_exact": o["optional_variation_exact"],
        "optional_variation_certificate": o["optional_variation_certificate"],
    }


def theorem_sweep(problem, eta_values: Sequence[float], optional_scale: float = 0.15):
    """Compact generic theorem sweep retained for backward compatibility."""
    rows = []
    thresholds = {
        "local": exact_local_threshold(problem, optional_scale),
        "global": exact_global_threshold(problem, optional_scale),
        "band": exact_band_threshold(problem, optional_scale),
    }
    for prop, threshold in thresholds.items():
        if threshold <= 0:
            continue
        for eta in eta_values:
            P = max(0.0, (1.0 + float(eta)) * threshold)
            rows.append({
                "seed": int(problem.seed), "n": int(problem.n), "k": int(problem.k),
                "property": prop, "oracle_threshold": float(threshold),
                "eta": float(eta), "penalty": float(P),
                "holds": int(property_holds(problem, prop, P, optional_scale)),
            })
    return rows


def theorem_plot_rows(problem, instance_index: int, optional_scale: float = 0.15):
    """Return old-schema rows for theorem_local/global/band/ordering plots."""
    idx = int(instance_index)
    pl = float(exact_local_threshold(problem, optional_scale))
    pg = float(exact_global_threshold(problem, optional_scale))
    pb = float(exact_band_threshold(problem, optional_scale))
    lcert = float(one_bit_lipschitz_certificate(problem, optional_scale))
    gcert = global_certificate(problem, optional_scale)
    bcert = float(coefficient_range_certificate(problem, optional_scale))
    exact_range = float(exact_base_range(problem, optional_scale))
    exact_gf = float(exact_infeasible_advantage(problem, optional_scale))

    local_rows, global_rows, band_rows = [], [], []
    for eta in THEOREM_PENALTY_ORACLE_ETA_RATIOS:
        if pl > 0.0:
            P = max(0.0, (1.0 + float(eta)) * pl)
            d = local_diagnostics(problem, P, optional_scale)
            local_rows.append({
                "instance_index": idx, "seed": int(problem.seed),
                "eta_l_oracle_ratio": float(eta), "penalty": P,
                "g_l": d["g_l"], "Gamma_l": d["Gamma_l"],
                # For this cardinality penalty a one-bit boundary violation is
                # exactly one, so exact gain and exact property boundary coincide.
                "exact_gain_eta_ratio": 0.0,
                "Delta_star_eta_ratio": _ratio_eta(lcert, pl),
            })
        if pg > 0.0:
            P = max(0.0, (1.0 + float(eta)) * pg)
            d = global_diagnostics(problem, P, optional_scale)
            global_rows.append({
                "instance_index": idx, "seed": int(problem.seed),
                "eta_g_oracle_ratio": float(eta), "penalty": P,
                "g_g": d["g_g"],
                "all_global_minimizers_feasible": d["all_global_minimizers_feasible"],
                "G_F_eta_ratio": _ratio_eta(exact_gf, pg),
                "Ghat_F_tight_eta_ratio": _ratio_eta(float(gcert["tight"]), pg),
                "Ghat_F_eta_ratio": _ratio_eta(float(gcert["value"]), pg),
            })
        if pb > 0.0:
            P = max(0.0, (1.0 + float(eta)) * pb)
            d = band_diagnostics(problem, P, optional_scale)
            band_rows.append({
                "instance_index": idx, "seed": int(problem.seed),
                "eta_u_oracle_ratio": float(eta), "penalty": P,
                "g_u": d["g_u"], "Gamma_u": d["Gamma_u"],
                "Delta_base_max_eta_ratio": _ratio_eta(exact_range, pb),
                "B_base_eta_ratio": _ratio_eta(bcert, pb),
            })

    o = ordering_certificates(problem)
    oracle = float(o["oracle"])
    ordering_rows = []
    if np.isfinite(oracle) and oracle > 0.0:
        cert_eta = _ratio_eta(float(o["exact_sufficient"]), oracle)
        practical_eta = _ratio_eta(float(o["practical"]), oracle)
        for eta in THEOREM_ORDER_ORACLE_ETA_RATIOS:
            lam = max(0.0, (1.0 + float(eta)) * oracle)
            d = ordering_diagnostics(problem, lam)
            ordering_rows.append({
                "instance_index": idx, "seed": int(problem.seed),
                "eta_lambda_oracle_ratio": float(eta), "lambda": lam,
                "g_ord": d["g_ord"], "Gamma_ord": d["Gamma_ord"],
                "kendall_tau": d["kendall_tau"], "pearson_r": d["pearson_r"],
                "lambda_cert_eta_ratio": cert_eta,
                "lambda_practical_eta_ratio": practical_eta,
            })
    return local_rows, global_rows, band_rows, ordering_rows


def ordering_sweep(problem, ratios: Sequence[float]):
    oracle = ordering_certificates(problem)["oracle"]
    rows = []
    if not np.isfinite(oracle) or oracle <= 0:
        return rows
    for ratio in ratios:
        lam = float(ratio) * oracle
        rows.append({
            "seed": int(problem.seed), "n": int(problem.n), "k": int(problem.k),
            "lambda_over_oracle": float(ratio), "lambda": float(lam),
            "ordering_oracle": float(oracle),
            "preserved": int(ordering_preserved(problem, lam)),
        })
    return rows
