"""Certificate-conservativeness experiment."""
from __future__ import annotations

from QUBO_Coefficient_Calibration.Experiment.analysis import (
    exact_local_threshold, exact_global_threshold, exact_band_threshold,
    one_bit_lipschitz_certificate, coefficient_range_certificate,
    global_certificate, ordering_certificates, exact_base_range,
    exact_infeasible_advantage,
)
from QUBO_Coefficient_Calibration.Experiment.optional import exact_thresholds

def conservativeness_row(problem, instance_index: int, optional_scale: float = 0.15):
    """One old-schema C1 row comparing exact boundaries and certificates."""
    g = global_certificate(problem, optional_scale)
    o = ordering_certificates(problem)
    opt, _ = exact_thresholds(problem)
    pl = float(exact_local_threshold(problem, optional_scale))
    return {
        "instance_index": int(instance_index), "seed": int(problem.seed),
        "N_x": int(problem.n), "n": int(problem.n), "k": int(problem.k),
        "P_local_property_boundary": pl,
        "Delta_local_exact_gain": pl,
        "Delta_star": float(one_bit_lipschitz_certificate(problem, optional_scale)),
        "P_global_property_boundary": float(exact_global_threshold(problem, optional_scale)),
        "G_F_exact": float(exact_infeasible_advantage(problem, optional_scale)),
        "Ghat_F_tight": float(g["tight"]),
        "Ghat_F": float(g["value"]),
        "P_band_property_boundary": float(exact_band_threshold(problem, optional_scale)),
        "Delta_base_max_exact": float(exact_base_range(problem, optional_scale)),
        "B_base": float(coefficient_range_certificate(problem, optional_scale)),
        "lambda_oracle": float(o["oracle"]),
        "lambda_cert": float(o["exact_sufficient"]),
        "lambda_practical": float(o["practical"]),
        "lambda_opt_property_boundary": float(opt["lambda_disp_exact"]),
        "lambda_opt_cert": float(opt["lambda_opt_cert"]),
        "lambda_opt_struct": float(opt["lambda_opt_struct"]),
        "finite_optimum_displacement": int(opt["finite_displacement_threshold"]),
    }
