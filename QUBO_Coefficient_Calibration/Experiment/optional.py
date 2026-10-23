"""Optional-term theory and QAOA experiments.

The exact threshold analysis, candidate construction, Nx study, and multiplier
study are kept together because they define one manuscript study family.
Returned QAOA solutions are always assessed using the primary objective.
"""
from __future__ import annotations

import math
from dataclasses import replace
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from QUBO_Coefficient_Calibration.Experiment.analysis import (
    enumerate_landscape, exact_global_threshold, global_certificate,
    ordering_certificates,
)
from QUBO_Coefficient_Calibration.Experiment.qaoa import _solve_multistart

# Ratio 1 is the SHARP ordering failure boundary, not a sufficient certificate.
DEFAULT_OPTIONAL_POWERS = (-3, -2, -1, 0, 1, 2, 3, 4)
DEFAULT_OPTIMA_RATIOS = (0., .25, .5, .75, .9, .99, 1., 1.01, 1.1, 1.25, 1.5, 2., 3., 4.)
OPTIONAL_ARMS = (
    ('ordering_tight', 'Ordering: exact / tight'),
    ('ordering_surrogate', 'Ordering: surrogate'),
    ('optimum_tight', 'Primary optimum: exact / tight'),
    ('optimum_surrogate', 'Primary optimum: surrogate'),
    ('beyond_displacement', 'Above displacement'),
)
CONSTRUCTIONS = ('random', 'marginal')

# Guarantee x calibration-type product used by the optional Nx study.
BOUND_ARMS = (
    ('ordering_tight', 'lambda_order_exact'),
    ('ordering_surrogate', 'lambda_order_practical'),
    ('optimum_tight', 'lambda_disp_exact'),
    ('optimum_surrogate', 'lambda_opt_struct'),
)

def cardinality_optional_lower_bound(problem) -> float:
    """L_k: k smallest optional linears plus choose(k,2) smallest pair terms.

    Include zeros for inactive quadratic pairs.  The independently minimized
    coefficients need not correspond to a realizable feasible configuration.
    """
    linear = np.sort(np.asarray(problem.optional_linear, dtype=float))
    pairs = np.array([float(problem.optional_quadratic.get((i, j), 0.0))
                      for i, j in combinations(range(problem.n), 2)], dtype=float)
    pair_count = int(problem.k * (problem.k - 1) // 2)
    return float(np.sum(linear[:problem.k]) +
                 (np.sum(np.sort(pairs)[:pair_count]) if pair_count else 0.0))


def _ratio(num: float, denom: float) -> float:
    if num == math.inf or denom <= 0.0:
        return math.inf
    return float(num / denom)


def exact_thresholds(problem) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute manuscript's five named scales and structural certificate.

    The two sufficient *enumerated* certificates need not be ordered against
    each other\'s exact property threshold.  In particular, lambda_opt_cert and
    lambda_order_exact are not generally comparable.
    """
    land = enumerate_landscape(problem, 0.0)
    feasible = np.asarray(land["feasible"], bool)
    pri = np.asarray(land["primary"][feasible], float)
    opt = np.asarray(land["optional_template"][feasible], float)
    bits = np.asarray(land["bits"][feasible], int)
    pstar = float(np.min(pri))
    primary_optimal = np.isclose(pri, pstar, rtol=0, atol=1e-10)
    suboptimal = ~primary_optimal
    ostar = float(np.min(opt[primary_optimal]))
    best_witness = int(np.flatnonzero(primary_optimal & np.isclose(opt, ostar, atol=1e-10, rtol=0))[0])
    optional_range = float(np.ptp(opt))

    # A robust, groupwise computation of the exact first strict-order tie.
    # For lower-primary group i and higher-primary group j, the first opposing
    # pair uses max optional in group i and min optional in group j.
    levels, groups = np.unique(pri, return_inverse=True)
    min_opt = np.array([np.min(opt[groups == g]) for g in range(len(levels))])
    max_opt = np.array([np.max(opt[groups == g]) for g in range(len(levels))])
    order_scale = math.inf
    order_witness = None
    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            opposing = float(max_opt[i] - min_opt[j])
            if opposing <= 1e-12:
                continue
            candidate = float((levels[j] - levels[i]) / opposing)
            if candidate < order_scale:
                low = int(np.flatnonzero((groups == i) & np.isclose(opt, max_opt[i], atol=1e-10))[0])
                high = int(np.flatnonzero((groups == j) & np.isclose(opt, min_opt[j], atol=1e-10))[0])
                order_scale = candidate
                order_witness = (low, high)

    # Pairwise primary gaps are positive differences between distinct levels.
    primary_min_gap = float(np.min(np.diff(levels))) if len(levels) > 1 else math.inf
    primary_opt_gap = float(np.min(pri[suboptimal] - pstar)) if np.any(suboptimal) else math.inf
    lam_order_cert = _ratio(primary_min_gap, optional_range)

    # Adverse advantage is relative to the *best optional value within the
    # whole primary-optimal set*, so primary ties are handled correctly.
    adverse = np.maximum(ostar - opt[suboptimal], 0.0)
    Aopt = float(np.max(adverse, initial=0.0))
    lam_opt_cert = _ratio(primary_opt_gap, Aopt)
    rivals = np.flatnonzero(suboptimal & (opt < ostar - 1e-12))
    if len(rivals):
        ratios = (pri[rivals] - pstar) / (ostar - opt[rivals])
        rival = int(rivals[int(np.argmin(ratios))])
        lam_disp = float(np.min(ratios))
    else:
        rival = None
        lam_disp = math.inf

    Lk = cardinality_optional_lower_bound(problem)
    if Lk > float(np.min(opt)) + 1e-9:
        raise AssertionError("L_k must lower-bound every feasible optional value")
    integer_primary = (
        np.allclose(problem.primary_linear, np.round(problem.primary_linear), atol=1e-10)
        and all(abs(c - round(c)) < 1e-10 for c in problem.primary_quadratic.values())
    )
    lam_opt_struct = _ratio(1.0, max(0.0, ostar - Lk)) if integer_primary else math.nan
    # Reuse the canonical coefficient-only full-ordering bound, but compare it
    # with our independently implemented vectorized exact thresholds.
    coeff_optional_L1 = float(np.sum(np.abs(problem.optional_linear)) +
                              sum(abs(v) for v in problem.optional_quadratic.values()))
    lam_order_practical = _ratio(1.0, coeff_optional_L1) if integer_primary else math.nan
    raw = {
        "seed": int(problem.seed), "n": int(problem.n), "k": int(problem.k),
        "feasible_count": int(len(pri)), "primary_optimal_count": int(np.count_nonzero(primary_optimal)),
        "primary_min_gap": primary_min_gap, "primary_optimum_gap": primary_opt_gap,
        "optional_feasible_range": optional_range, "adverse_optional_advantage": Aopt,
        "best_primary_optional": ostar, "optional_structural_lower_bound": Lk,
        "primary_coefficients_integer": int(integer_primary),
        "lambda_order_practical": lam_order_practical,
        "lambda_order_cert": lam_order_cert,
        "lambda_order_exact": float(order_scale),
        "lambda_opt_cert": lam_opt_cert,
        "lambda_opt_struct": lam_opt_struct,
        "lambda_disp_exact": lam_disp,
        "finite_order_threshold": int(math.isfinite(order_scale)),
        "finite_displacement_threshold": int(math.isfinite(lam_disp)),
        "order_witness_low": _bits(bits, order_witness[0]) if order_witness else "",
        "order_witness_high": _bits(bits, order_witness[1]) if order_witness else "",
        "displacement_witness_optimum": _bits(bits, best_witness),
        "displacement_witness_rival": _bits(bits, rival) if rival is not None else "",
    }
    for key in ("lambda_order_exact", "lambda_opt_cert", "lambda_opt_struct", "lambda_disp_exact"):
        if math.isfinite(raw[key]) and math.isfinite(lam_order_cert) and lam_order_cert > 0:
            raw[key + "_over_order_cert"] = float(raw[key] / lam_order_cert)
        else:
            raw[key + "_over_order_cert"] = math.inf
    raw["lambda_order_practical_over_exact"] = (
        float(lam_order_practical / order_scale) if math.isfinite(order_scale) and order_scale > 0
        else math.nan
    )
    raw["lambda_order_cert_over_exact"] = (
        float(lam_order_cert / order_scale) if math.isfinite(order_scale) and order_scale > 0
        else math.nan
    )
    raw["lambda_opt_cert_over_exact"] = (
        float(lam_opt_cert / lam_disp) if math.isfinite(lam_disp) and lam_disp > 0
        else math.nan
    )
    raw["lambda_opt_struct_over_exact"] = (
        float(lam_opt_struct / lam_disp) if math.isfinite(lam_disp) and lam_disp > 0
        else math.nan
    )
    context = dict(primary=pri, optional=opt, bits=bits, optimal=primary_optimal,
                   order_witness=order_witness, disp_rival=rival,
                   primary_opt_witness=best_witness)
    return raw, context


def _bits(bitstrings: np.ndarray, index: int) -> str:
    return "".join(map(str, bitstrings[index]))


def exact_property_status(context: dict, lam: float) -> tuple[bool, bool]:
    """Independent enumerated tests of both properties, tolerant to roundoff.

    The strict boundaries themselves are handled by the analytic witnesses,
    not numerically classified using tolerance at a floating-point tie.
    """
    pri, opt = context["primary"], context["optional"]
    values = pri + float(lam) * opt
    # Sort/group by exact primary value; only distinct group comparisons count.
    levels, groups = np.unique(pri, return_inverse=True)
    lo = np.array([np.min(values[groups == i]) for i in range(len(levels))])
    hi = np.array([np.max(values[groups == i]) for i in range(len(levels))])
    scale = 1.0 + float(np.max(np.abs(values)))
    ordering = bool(np.all(hi[:-1] < lo[1:] - 1e-9 * scale))
    optimum = context["optimal"]
    # The guarantee is about EVERY perturbed minimizer.  A tied suboptimal
    # state at the minimum makes it false.
    good = float(np.min(values[optimum]))
    bad = float(np.min(values[~optimum])) if np.any(~optimum) else math.inf
    return ordering, bool(good < bad - 1e-9 * scale)


def theory_instance(problem, instance_index: int, *, boundary_offset: float = 0.02
                    ) -> tuple[dict, list[dict]]:
    """Per-instance threshold table and independently checked boundary probes."""
    if not (0 < boundary_offset < .5):
        raise ValueError("boundary_offset must be in (0, 0.5)")
    t, c = exact_thresholds(problem)
    t["instance"] = int(instance_index)
    if not t["primary_coefficients_integer"]:
        raise ValueError("Structural case study requires integer primary coefficients")
    lc, lo, oc, d = (t[k] for k in ("lambda_order_cert", "lambda_order_exact",
                                  "lambda_opt_cert", "lambda_disp_exact"))
    finite = lambda v: math.isfinite(v) and v > 0
    # The two partial order relations are deliberate.  NEVER assume that
    # lambda_opt_cert <= or >= lambda_order_exact.
    checks = [
        ("order_certificate_below_exact", lc, lo),
        ("ordering_below_displacement", lo, d),
        ("order_certificate_below_opt_certificate", lc, oc),
        ("opt_certificate_below_displacement", oc, d),
        ("structural_below_displacement", t["lambda_opt_struct"], d),
        ("practical_below_order_certificate", t["lambda_order_practical"], lc),
    ]
    for name, low, upper in checks:
        if math.isfinite(low) and math.isfinite(upper) and low > upper + 1e-9 * max(1., upper):
            raise AssertionError(f"Hierarchy failed: {name} for seed={problem.seed}: {low}>{upper}")
        t["verified_" + name] = 1
    canonical = ordering_certificates(problem)
    if not _same_threshold(t["lambda_order_cert"], canonical["exact_sufficient"]) or not _same_threshold(lo, canonical["oracle"]):
        raise AssertionError("Independent ordering scales disagree with existing implementation")
    if not _same_threshold(d, _independent_displacement(c)):
        raise AssertionError("Displacement ratio disagrees with independent exhaustive check")

    probes = []
    def add(label, reference, ratio, lam, expected_order=None, expected_opt=None,
            boundary_witness=None):
        order, optimum = exact_property_status(c, lam)
        if expected_order is not None and order != expected_order:
            raise AssertionError(f"{label}: strict ordering expected {expected_order}, observed {order}")
        if expected_opt is not None and optimum != expected_opt:
            raise AssertionError(f"{label}: primary optimality expected {expected_opt}, observed {optimum}")
        witness_gap = math.nan
        if boundary_witness is not None:
            low, high = boundary_witness
            pri, opt = c["primary"], c["optional"]
            witness_gap = float((pri[high] + lam * opt[high]) - (pri[low] + lam * opt[low]))
            if ratio == 1.0 and abs(witness_gap) > 1e-8 * (1 + abs(pri[low]) + abs(pri[high]) + abs(lam)*(abs(opt[low])+abs(opt[high]))):
                raise AssertionError(f"Boundary witness not tied: {label}: {witness_gap}")
        probes.append({"instance": instance_index, "seed": int(problem.seed),
                       "n": int(problem.n), "k": int(problem.k),
                       "probe": label, "reference": reference, "ratio": ratio,
                       "lambda": lam, "strict_ordering_preserved": int(order),
                       "every_minimizer_primary_optimal": int(optimum),
                       "analytic_boundary_tie": int(ratio == 1.0 and boundary_witness is not None),
                       "boundary_witness_signed_gap": witness_gap})

    add("unperturbed", "zero", 0.0, 0., expected_order=True, expected_opt=True)
    if finite(lc):
        add("order_cert_inside", "lambda_order_cert", 1-boundary_offset,
            (1-boundary_offset)*lc, expected_order=True, expected_opt=True)
    if finite(oc):
        add("opt_cert_inside", "lambda_opt_cert", 1-boundary_offset,
            (1-boundary_offset)*oc, expected_opt=True)
    sc = t["lambda_opt_struct"]
    if finite(sc):
        add("opt_struct_inside", "lambda_opt_struct", 1-boundary_offset,
            (1-boundary_offset)*sc, expected_opt=True)
    if finite(lo):
        low, high = c["order_witness"]
        for ratio in (1-boundary_offset, 1., 1+boundary_offset):
            name = "order_below" if ratio < 1 else "order_equal" if ratio == 1 else "order_above"
            add(name, "lambda_order_exact", ratio, ratio*lo,
                expected_order=True if ratio < 1 else None if ratio == 1 else False,
                expected_opt=True if ratio < 1 else None,
                boundary_witness=(low, high))
    if finite(d):
        # Equality pairs the primary optimum with the first offending rival.
        best, rival = c["primary_opt_witness"], c["disp_rival"]
        for ratio in (1-boundary_offset, 1., 1+boundary_offset):
            name = "disp_below" if ratio < 1 else "disp_equal" if ratio == 1 else "disp_above"
            add(name, "lambda_disp_exact", ratio, ratio*d,
                expected_opt=True if ratio < 1 else None if ratio == 1 else False,
                boundary_witness=(best, rival))
    else:
        t["infinite_displacement_note"] = "No primary-suboptimal state has optional advantage over the best primary optimum"
    t["boundary_probe_count"] = len(probes)
    t["all_checks_passed"] = 1
    return t, probes


def _same_threshold(left: float, right: float) -> bool:
    return left == right if math.isinf(left) or math.isinf(right) else math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-10)


def _independent_displacement(context) -> float:
    p, o, primary_mask = context["primary"], context["optional"], context["optimal"]
    best, base_optional = float(np.min(p)), float(np.min(o[primary_mask]))
    rivals = (~primary_mask) & (o < base_optional - 1e-12)
    return (float(np.min((p[rivals]-best)/(base_optional-o[rivals])))
            if np.any(rivals) else math.inf)

def marginal_cost_problem(problem, max_linear=2.0):
    """Build a *non-oracular* marginal-cost optional template.

    For selecting k out of n variables, the expected number of selected
    partners for a selected variable is k-1 among n-1 other variables.  Thus
    s_i = a_i + (k-1)/(n-1) * sum_{j != i} b_ij is a simple mean-field
    approximation to its selection cost.  We center s (a constant shift on
    the fixed-cardinality feasible set) and normalize the largest absolute
    entry to ``max_linear``.  The optional template is linear, not an exact
    reformulation of H_primary, and requires NO feasible enumeration,
    optimal solution, learned parameters, or QAOA outcome to construct.

    We deliberately do not carry over the random optional quadratic term:
    that would obscure whether marginal-cost guidance itself has an effect.
    Normalization is coefficient-only, not calibrated using exact solutions.
    Both constructions subsequently receive their own exact/surrogate bounds.
    """
    n, k = int(problem.n), int(problem.k)
    if not (math.isfinite(max_linear) and max_linear > 0):
        raise ValueError('marginal max linear magnitude must be finite and positive')
    incident = np.zeros(n, dtype=float)
    for (i, j), value in problem.primary_quadratic.items():
        incident[i] += float(value)
        incident[j] += float(value)
    scores = np.asarray(problem.primary_linear, dtype=float) + (
        (k - 1) / (n - 1) * incident
    )
    centered = scores - float(np.mean(scores))
    radius = float(np.max(np.abs(centered)))
    if radius > 1e-12:
        centered = centered * (float(max_linear) / radius)
    else:
        centered = np.zeros(n, dtype=float)
    return replace(problem, optional_linear=centered, optional_quadratic={})


def _construction_problems(random_problem, constructions, marginal_max_linear):
    names = tuple(dict.fromkeys(constructions))
    if not names or any(name not in CONSTRUCTIONS for name in names):
        raise ValueError('constructions must be a nonempty subset of random,marginal')
    return {
        name: (random_problem if name == 'random' else
               marginal_cost_problem(random_problem, max_linear=marginal_max_linear))
        for name in names
    }


def optimum_theorem_rows(problem, instance: int, ratios: Sequence[float] = DEFAULT_OPTIMA_RATIOS):
    """Two theorem diagnostics at lambda/lambda_disp, including an explicit tie.

    A primary-suboptimal state is counted as an adverse competitor only if its
    perturbed energy is STRICTLY below the best perturbed energy on the original
    primary-optimal set.  At lambda_disp an exact tie violates 'EVERY minimizer'
    even though this strict competitor count can still be zero.
    """
    t, ctx = exact_thresholds(problem)
    audit = {'instance': int(instance), **t}
    boundary = float(t['lambda_disp_exact'])
    if not math.isfinite(boundary) or boundary <= 0:
        audit['optima_sweep_status'] = 'unbounded_or_zero_displacement; excluded from normalized sweep'
        return audit, []

    primary = np.asarray(ctx['primary'], float)
    optional = np.asarray(ctx['optional'], float)
    mask = np.asarray(ctx['optimal'], bool)
    orig = float(primary[mask][0])
    n_sub = int(np.count_nonzero(~mask))
    rows = []
    for ratio in sorted(set(float(x) for x in ratios)):
        if ratio < 0:
            raise ValueError('optimum theorem ratios must be nonnegative')
        lam = float(ratio * boundary)
        base = primary + lam * optional
        p_opt_reference = float(np.min(base[mask]))
        winning = float(np.min(base))
        # Scale-aware tie tolerance; at the exact analytical boundary the tie is
        # present even if binary floating-point subtraction errs slightly.
        tol = 1e-9 * max(1., np.max(np.abs(base)), abs(lam))
        minimizers = base <= winning + tol
        worst_primary_gap = max(0., float(np.max(primary[minimizers])) - orig)
        strict_competitors = int(np.count_nonzero((~mask) & (base < p_opt_reference - tol)))
        tied_competitors = int(np.count_nonzero((~mask) & (np.abs(base - p_opt_reference) <= tol)))
        # The exact ratio theorem decides equality even under floating noise.
        if np.isclose(ratio, 1., atol=1e-12, rtol=0.0):
            if tied_competitors == 0:
                raise AssertionError('Displacement boundary must have a suboptimal tying witness')
        rows.append({
            'instance': int(instance), 'seed': int(problem.seed), 'n': int(problem.n),
            'k': int(problem.k), 'lambda': lam, 'lambda_over_disp': ratio,
            'lambda_disp_exact': boundary,
            'lambda_order_over_disp': float(t['lambda_order_exact']) / boundary,
            'lambda_order_cert_over_disp': float(t['lambda_order_cert']) / boundary,
            'lambda_opt_cert_over_disp': float(t['lambda_opt_cert']) / boundary,
            'lambda_opt_struct_over_disp': float(t['lambda_opt_struct']) / boundary,
            'primary_optimum': orig, 'worst_global_minimizer_primary_regret': worst_primary_gap,
            'every_minimizer_primary_optimal': int(worst_primary_gap <= tol),
            'strictly_better_suboptimal_count': strict_competitors,
            'strictly_better_suboptimal_percent': 100. * strict_competitors / max(1, n_sub),
            'tied_suboptimal_count': tied_competitors,
            'original_primary_optimum_perturbed_energy': p_opt_reference,
            'min_perturbed_feasible_energy': winning,
        })
    audit['optima_sweep_status'] = 'included'
    return audit, rows


def expected_best_feasible_primary(probabilities, landscape, shots: int):
    """Exact expected primary value of the best FEASIBLE draw in `shots` shots.

    The result is conditional on observing >=1 feasible draw.  Infeasible draws
    are ignored; neither optional energies nor modified-Hamiltonian values enter
    the postprocessing.  No Monte Carlo noise or independent-shot pseudo-reps.
    """
    if shots < 1:
        raise ValueError('shots must be positive')
    probs = np.asarray(probabilities, float)
    feas = np.asarray(landscape['feasible'], bool)
    primary = np.asarray(landscape['primary'], float)
    if len(probs) != len(feas):
        raise ValueError('probability vector and landscape have different lengths')
    levels = np.unique(primary[feas])
    p_feas = float(np.sum(probs[feas]))
    have = -math.expm1(shots * math.log1p(-min(p_feas, 1.))) if p_feas < 1. else 1.
    if have <= 1e-15:
        return math.nan, have
    prev = 0.
    total = 0.
    for level in levels:
        current = float(np.sum(probs[feas & (primary <= level + 1e-10)]))
        current = min(1., max(prev, current))
        p_best_this = max(0., (1.-prev)**shots - (1.-current)**shots)
        total += float(level) * p_best_this
        prev = current
    return float(total / have), float(have)


def _calibration(problem):
    thresholds, _ = exact_thresholds(problem)
    return thresholds


def _valid_scale(x):
    return math.isfinite(float(x)) and float(x) > 0.


def _fixed_feasibility_penalty(problem, lambdas, basis, safety):
    if safety <= 1.:
        raise ValueError('feasibility safety factor must exceed 1')
    if basis == 'exact-global':
        thresholds = [exact_global_threshold(problem, lam) for lam in lambdas]
    elif basis == 'certified-global':
        thresholds = [global_certificate(problem, lam)['tight'] for lam in lambdas]
    else:
        raise ValueError('unknown penalty basis')
    exact = [exact_global_threshold(problem, lam) for lam in lambdas]
    return max(1e-5, safety * max(thresholds + exact + [1e-8])), max(exact)


def _qaoa_rows(problem, instance: int, arms, *, penalty_basis='exact-global',
               penalty_factor=1.05, shots=32, p=3, restarts=3, maxiter=100,
               stepsize=.05, optimizer='adam', engine='pennylane',
               device_name='default.qubit', seed_base=12345,
               construction='random', shared_penalty=None, solved_cache=None):
    """Shared penalty, same restart seeds and primary-only evaluation per arm."""
    arms = [(str(label), float(lam), float(ratio)) for label, lam, ratio in arms]
    if any((not math.isfinite(lam) or lam < 0) for _,lam,_ in arms):
        raise ValueError('QAOA requires nonnegative finite optional coefficients')
    if shared_penalty is None:
        pen, required = _fixed_feasibility_penalty(
            problem, [lam for _, lam, _ in arms], penalty_basis, penalty_factor
        )
    else:
        pen, required = shared_penalty
    land = enumerate_landscape(problem, optional_scale=0.)
    orig = float(np.min(land['primary'][land['feasible']]))
    rows = []
    solved = {} if solved_cache is None else solved_cache
    for label, lam, ratio in arms:
        # The lambda=0 Hamiltonian is identical for both constructions; all
        # other cache keys explicitly include the optional construction.
        key = ('baseline', 0.) if lam == 0. else (construction, lam)
        if key not in solved:
            solved[key] = _solve_multistart(
                problem, penalty=pen, optional_scale=lam, p=p, restarts=restarts,
                maxiter=maxiter, stepsize=stepsize, optimizer=optimizer,
                device_name=device_name, seed_base=seed_base, engine=engine,
                seed_stride=131,
            )
        _, result, metrics, qseed, restart, hist = solved[key]
        best, usable = expected_best_feasible_primary(result['final_probabilities'], land, shots)
        row = {
            'instance': int(instance), 'seed': int(problem.seed), 'N_x': int(problem.n),
            'n': int(problem.n), 'k': int(problem.k), 'arm': label,
            'construction': 'baseline' if label == 'baseline' else construction,
            'lambda': lam, 'lambda_multiplier': ratio,
            'fixed_penalty': pen, 'exact_feasibility_boundary_max': required,
            'penalty_basis': penalty_basis, 'penalty_factor': penalty_factor,
            'qaoa_depth': int(p), 'qaoa_restarts': int(restarts),
            'qaoa_optimizer': optimizer if engine == 'pennylane' else 'lbfgs',
            'qaoa_iterations': int(maxiter), 'qaoa_engine': engine,
            'qaoa_device': device_name, 'shots': int(shots),
            'best_feasible_32shot_primary': best,  # retained key for generic plots
            'best_feasible_shot_primary': best,
            'best_feasible_shot_regret_percent':
                100.*(best-orig)/max(1.,abs(orig)) if math.isfinite(best) else math.nan,
            'prob_at_least_one_feasible': usable,
            'p_best_feasible_single_shot': float(metrics['p_primary_optimum']),
            'p_best_feasible_in_shots': 1.-(1.-float(metrics['p_primary_optimum']))**int(shots),
            'p_feasible_single_shot': float(metrics['p_feasible']),
            'primary_optimum': orig, 'selected_restart': int(restart),
            'selected_qaoa_seed': int(qseed), 'qaoa_final_expval': float(result['expval_energy_optimization_scale']),
            'qaoa_iterations_completed': int(len(hist)),
        }
        rows.append(row)
    return rows


def _five_arm_bounds(t):
    """Return all four finite, positive bounds or explain exclusions.

    The coefficient-only optimum surrogate assumes integer primary coefficients
    and a certified original primary optimum; this case study enumerates that
    witness, as disclosed in optional.py.
    """
    missing = [key for _, key in BOUND_ARMS if not _valid_scale(t[key])]
    return missing


def optional_candidate_screen(problem, constructions=('random',), marginal_max_linear=2.):
    """Exact, QAOA-independent acceptance test for a MATCHED optional cohort.

    All four calibration bounds must be finite and positive for every chosen
    construction.  In particular a genuine infinite displacement threshold is
    rejected, never assigned an arbitrary large coefficient.  The generator
    searches subsequent independently seeded candidates until the user-specified
    number of ACCEPTED primary instances is reached.
    """
    problems = _construction_problems(problem, constructions, marginal_max_linear)
    row = {'seed': int(problem.seed), 'n': int(problem.n),
           'k': int(problem.k), 'constructions': ','.join(problems)}
    exclusions = []
    for name, current in problems.items():
        bounds = _calibration(current)
        missing = _five_arm_bounds(bounds)
        row.update({f'{name}_{key}': val for key, val in bounds.items()
                    if key in ('lambda_order_exact','lambda_order_practical',
                               'lambda_disp_exact','lambda_opt_struct')})
        row[f'{name}_eligibility'] = 'eligible' if not missing else 'ineligible: ' + ','.join(missing)
        if missing:
            exclusions.append(name + ':' + ','.join(missing))
    row['accepted'] = int(not exclusions)
    row['rejection_reason'] = ';'.join(exclusions)
    return row


def _shared_template_penalty(plans, *, penalty_basis, penalty_factor):
    """One P for the original primary-only baseline and BOTH constructions."""
    if not (math.isfinite(penalty_factor) and penalty_factor > 1.):
        raise ValueError('shared-penalty safety factor must be finite and > 1')
    if not plans:
        raise ValueError('cannot calibrate a shared penalty with no policies')
    candidates, exact = [], []
    # Crucial: the zero-optional baseline is ALSO a tested Hamiltonian.
    # Feasibility thresholds need not be monotone in lambda, so include P(0)
    # even if every listed optional policy happens to require a lower P.
    first_problem = plans[0][1]
    baseline_exact = exact_global_threshold(first_problem, 0.)
    exact.append(baseline_exact)
    candidates.append(baseline_exact if penalty_basis == 'exact-global' else
                      global_certificate(first_problem, 0.)['tight'])
    for _, problem, arms, _ in plans:
        for _, lam, _ in arms:
            required = exact_global_threshold(problem, lam)
            exact.append(required)
            candidates.append(required if penalty_basis == 'exact-global' else
                              global_certificate(problem, lam)['tight'])
    return (max(1e-5, penalty_factor * max(candidates + exact + [1e-8])),
            max(exact + [0.]))


def _run_constructions(plans, instance, **qaoa):
    """Evaluate paired coefficient plans using exactly one common penalty."""
    qaoa = dict(qaoa)
    penalty = _shared_template_penalty(
        plans,
        penalty_basis=qaoa.get('penalty_basis', 'exact-global'),
        penalty_factor=qaoa.get('penalty_factor', 1.05),
    )
    cache, rows = {}, []
    # Always run the primary-only control, even when only one construction is
    # selected.  Baseline is evaluated exactly once per problem/seed/budget.
    first_name, first_problem, _, _ = plans[0]
    baseline = _qaoa_rows(first_problem, instance, [('baseline', 0., 0.)],
                          construction=first_name, shared_penalty=penalty,
                          solved_cache=cache, **qaoa)
    rows.extend(baseline)
    for construction, problem, arms, metadata in plans:
        new = _qaoa_rows(problem, instance, arms, construction=construction,
                         shared_penalty=penalty, solved_cache=cache, **qaoa)
        if len(new) != len(metadata):
            raise AssertionError('Each optional arm requires one QAOA row')
        for row, info in zip(new, metadata):
            row.update(info)
        rows.extend(new)
    # The real solver rows always record fixed_penalty.  Synthetic test rows
    # may deliberately contain only the observed metrics/arms.
    recorded = [r['fixed_penalty'] for r in rows if 'fixed_penalty' in r]
    if recorded and not all(np.isclose(v, recorded[0]) for v in recorded):
        raise AssertionError('Paired optional experiments require identical penalties')
    return rows


def optional_nx_instance(problem, instance: int, *, exact_fraction=.95,
                         surrogate_fraction=.95, optimum_fraction=.95,
                         above_fraction=1.05, constructions=('random',),
                         marginal_max_linear=2., **qaoa):
    """Paired construction comparison; baseline + five policies per template.

    All five arms have matched starts and ONE penalty that protects global
    feasibility across their entire tested coefficient range.
    """
    problems = _construction_problems(problem, constructions, marginal_max_linear)
    audit = {'instance': int(instance), 'n': int(problem.n), 'seed': int(problem.seed),
             'constructions': ','.join(problems),
             'marginal_max_linear': float(marginal_max_linear)}
    for method, current in problems.items():
        audit[f'{method}_optional_linear'] = ','.join(f'{v:.12g}' for v in current.optional_linear)
    if not (0 < exact_fraction < 1 and 0 < surrogate_fraction < 1
            and 0 < optimum_fraction < 1 and above_fraction > 1):
        raise ValueError('Nx safe fractions must be in (0,1), unsafe factor >1')
    plans, exclusions = [], []
    for construction, current in problems.items():
        t = _calibration(current)
        for key, val in t.items():
            audit[f'{construction}_{key}'] = val
        missing = _five_arm_bounds(t)
        audit[f'{construction}_eligibility'] = 'eligible' if not missing else (
            'ineligible: ' + ','.join(missing))
        if missing:
            exclusions.append(construction + ':' + ','.join(missing))
            continue
        order, disp = float(t['lambda_order_exact']), float(t['lambda_disp_exact'])
        names = [
            ('ordering_tight', exact_fraction * order),
            ('ordering_surrogate', surrogate_fraction * float(t['lambda_order_practical'])),
            ('optimum_tight', optimum_fraction * disp),
            ('optimum_surrogate', optimum_fraction * float(t['lambda_opt_struct'])),
            ('beyond_displacement', above_fraction * disp),
        ]
        tol = 1e-8 * max(1., order, disp)
        if (names[0][1] >= order or names[1][1] >= order + tol or
                names[2][1] >= disp or names[3][1] >= disp + tol or
                names[4][1] <= disp):
            raise AssertionError('A calibrated optional arm violates its guaranteed interval')
        audit[f'{construction}_coefficients'] = {name: float(lam) for name, lam in names}
        metadata = [dict(lambda_order_exact=order,
                         lambda_order_practical=float(t['lambda_order_practical']),
                         lambda_disp_exact=disp,
                         lambda_opt_struct=float(t['lambda_opt_struct'])) for _ in names]
        plans.append((construction, current,
                      [(name, lam, lam / order) for name, lam in names], metadata))
    if exclusions:
        audit['optional_nx_status'] = 'excluded from paired cohort: ' + ';'.join(exclusions)
        audit['optional_nx_arm_count'] = 0
        return audit, []
    audit.update(optional_nx_status='included',
                 optional_nx_arm_count=1 + 5*len(plans))
    if len(plans) == 1:
        # Keep convenient old audit fields for single-construction use.
        t = _calibration(plans[0][1]);audit.update(t)
        audit['optional_nx_coefficients'] = audit[f'{plans[0][0]}_coefficients']
    rows = _run_constructions(plans, instance, **qaoa)
    return audit, rows


def optional_multiplier_instance(problem, instance: int, *,
                                 powers=DEFAULT_OPTIONAL_POWERS,
                                 include_zero=True, mode='ordering-tight',
                                 above_fraction=1.05, constructions=('random',),
                                 marginal_max_linear=2., **qaoa):
    """One exact-ordering-normalized sweep per optional construction.

    Both random and marginal constructions use their own sharp ordering
    threshold as the unit: lambda = 2**power * lambda_order_exact.  This
    compares *relative calibration rules*, not equal absolute perturbations.
    Other thresholds remain in each row for figure reference lines, but are
    not additional QAOA sweep arms.  The unused ``above_fraction`` keyword
    is accepted for backwards compatibility with old runner calls.

    As with the size study, a complete matched cohort requires finite,
    positive exact ordering and displacement thresholds and structural
    certificates for EVERY selected construction.  The integrated runner
    regenerates rejected instances before any QAOA evaluation.
    """
    if mode not in ('ordering-tight', 'common'):
        raise ValueError('mode must be ordering-tight (common is a legacy alias)')
    problems = _construction_problems(problem, constructions, marginal_max_linear)
    audit = {'instance': int(instance), 'n': int(problem.n), 'seed': int(problem.seed),
             'constructions': ','.join(problems),
             'marginal_max_linear': float(marginal_max_linear),
             'optional_multiplier_mode': mode}
    for method, current in problems.items():
        audit[f'{method}_optional_linear'] = ','.join(f'{v:.12g}' for v in current.optional_linear)
    powers = sorted(set(int(v) for v in powers))
    if not powers:
        raise ValueError('multiplier powers must not be empty')
    plans, exclusions = [], []
    for construction, current in problems.items():
        t = _calibration(current)
        for key, val in t.items(): audit[f'{construction}_{key}'] = val
        order = float(t['lambda_order_exact'])
        missing = _five_arm_bounds(t)
        audit[f'{construction}_eligibility'] = 'eligible' if not missing else (
            'ineligible: ' + ','.join(missing))
        if missing:
            exclusions.append(construction + ':' + ','.join(missing))
            continue
        arms, metadata = [], []
        for e in powers:
            multiplier = 2. ** e
            arms.append(('ordering_tight', order * multiplier, multiplier))
            metadata.append(('lambda_order_exact', multiplier))
        column_metadata = [dict(
            sweep_mode='ordering-tight', reference_bound=ref, bound_multiplier=bm,
            lambda_order_exact=order,
            lambda_order_cert_over_order=float(t['lambda_order_cert'])/order,
            lambda_order_practical_over_order=float(t['lambda_order_practical'])/order,
            lambda_opt_cert_over_order=float(t['lambda_opt_cert'])/order,
            lambda_opt_struct_over_order=float(t['lambda_opt_struct'])/order,
            lambda_disp_exact_over_order=float(t['lambda_disp_exact'])/order,
            finite_displacement=int(math.isfinite(float(t['lambda_disp_exact']))),
        ) for ref, bm in metadata]
        plans.append((construction, current, arms, column_metadata))
    if exclusions:
        audit['optional_multiplier_status']='excluded from paired cohort: '+';'.join(exclusions)
        audit['optional_multiplier_arm_count']=0
        return audit,[]
    audit.update(optional_multiplier_status='included',
                 optional_multiplier_arm_count=sum(len(plan[2]) for plan in plans) + int(include_zero),
                 optional_multiplier_powers=','.join(map(str,powers)),
                 optional_multiplier_reference='lambda_order_exact')
    rows = _run_constructions(plans, instance, **qaoa)
    if not include_zero:
        rows = [r for r in rows if r['arm'] != 'baseline']
    else:
        # Mode is global to the experiment; unlike threshold annotations, it
        # remains meaningful for the shared primary-only baseline row.
        for row in rows:
            if row['arm'] == 'baseline':
                row['sweep_mode'] = 'ordering-tight'
    if len(rows) != audit['optional_multiplier_arm_count']:
        raise AssertionError('QAOA results must contain every ordering-tight multiplier arm')
    if len(plans) == 1:audit.update(_calibration(plans[0][1]))
    return audit, rows


def paired_construction_rows(rows, *, multiplier=False):
    """Strict same-instance policy comparisons (marginal minus random).

    In the ordering-tight multiplier experiment, two constructions at the
    same nominal power refer to their OWN exact ordering threshold.  Save
    both ACTUAL lambda values: this is a contrast between equally scaled
    calibration policies, not between equal absolute coefficients.
    """
    keys = ('instance', 'n', 'arm', 'bound_multiplier') if multiplier else (
        'instance', 'n', 'arm')
    matched = {}
    for row in rows:
        method = row.get('construction')
        if method not in ('random', 'marginal'):
            continue
        index = tuple(row.get(k) for k in keys)
        matched.setdefault(index, {})[method] = row
    contrasts = []
    for key, family in sorted(matched.items(), key=lambda v: str(v[0])):
        if set(family) != {'random', 'marginal'}:
            continue
        rand, marg = family['random'], family['marginal']
        if (rand['seed'] != marg['seed'] or
                not np.isclose(rand['fixed_penalty'], marg['fixed_penalty'])):
            raise AssertionError('A paired construction contrast requires identical problem and penalty')
        item = dict(zip(keys, key))
        item.update(seed=rand['seed'], fixed_penalty=rand['fixed_penalty'],
                    random_lambda=rand['lambda'], marginal_lambda=marg['lambda'],
                    random_lambda_over_order=rand['lambda_multiplier'],
                    marginal_lambda_over_order=marg['lambda_multiplier'])
        for field in ('p_best_feasible_single_shot',
                      'best_feasible_shot_primary', 'best_feasible_shot_regret_percent'):
            x, y = float(rand[field]), float(marg[field])
            item[f'random_{field}'] = x
            item[f'marginal_{field}'] = y
            item[f'marginal_minus_random_{field}'] = y-x
        contrasts.append(item)
    return contrasts
