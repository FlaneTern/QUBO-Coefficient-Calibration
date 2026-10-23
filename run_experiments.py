#!/usr/bin/env python3
"""Run the theory-facing fixed-cardinality QUBO experiments.

CPU parallelism
---------------
Independent QAOA problem instances are executed in separate spawned worker
processes.  Use ``--max-parallel-instances N`` to bound concurrent workers;
``0`` selects an automatic value (up to four CPU workers, one GPU worker).

GPU QAOA
--------
``--compute-device gpu`` maps PennyLane statevector QAOA to ``lightning.gpu``.
``--compute-device auto`` uses that plugin when installed and otherwise falls
back to CPU.  ``--qaoa-device`` can override the automatic device selection.
"""
from __future__ import annotations

# IMPORTANT: native thread counts must be set before NumPy/PennyLane/Lightning
# are imported in the main process. Spawned workers inherit these values.
import os
import sys

def _early_threads_from_argv(default: int = 1) -> int:
    # Spawned multiprocessing workers may not receive the parent's CLI flags,
    # but DO inherit its deliberately configured native-thread environment.
    try:
        value = int(os.environ.get('OMP_NUM_THREADS', str(default)))
    except (ValueError, TypeError):
        value = default
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--threads-per-worker" and i + 1 < len(argv):
            try:
                value = int(argv[i + 1])
            except ValueError:
                pass
        elif arg.startswith("--threads-per-worker="):
            try:
                value = int(arg.split("=", 1)[1])
            except ValueError:
                pass
    return max(1, int(value))

def _apply_native_thread_env(n: int) -> None:
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n)
    # Keep native runtimes from silently changing the requested team size.
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")

_EARLY_THREADS_PER_WORKER = _early_threads_from_argv(1)
_apply_native_thread_env(_EARLY_THREADS_PER_WORKER)

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence, Any

import numpy as np

from QUBO_Coefficient_Calibration.Problem.cardinality import build_random_cardinality_problem
from QUBO_Coefficient_Calibration.Experiment.theorem import (
    theorem_summary, theorem_sweep, ordering_sweep, theorem_plot_rows,
)
from QUBO_Coefficient_Calibration.Experiment.conservativeness import conservativeness_row
from QUBO_Coefficient_Calibration.Experiment.spectral import run_spectral_instance, SPECTRAL_P_OVER_GLOBAL
from QUBO_Coefficient_Calibration.Experiment.penalty import (
    run_qaoa_penalty_instance, run_qaoa_scaling_instance, QAOA_P_OVER_GLOBAL,
)
from QUBO_Coefficient_Calibration.Experiment.budget import (
    run_qaoa_budget_instance, BUDGET_P_OVER_GLOBAL, DEFAULT_WORK_BUDGETS,
)


def write_csv(path: Path, rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys, seen = [], set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def _parse_int_list(text: str) -> list[int]:
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def _cpu_lightning_available() -> bool:
    try:
        return importlib.util.find_spec("pennylane_lightning") is not None
    except Exception:
        return False


def _gpu_plugin_available() -> bool:
    try:
        return (
            importlib.util.find_spec("pennylane_lightning") is not None
            and importlib.util.find_spec("pennylane_lightning.lightning_gpu") is not None
        )
    except Exception:
        return False


def _resolve_device(compute_device: str, qaoa_device: str, qaoa_engine: str, gpu_device_id: int):
    requested = str(compute_device).lower().strip()
    if requested not in {"cpu", "gpu", "auto"}:
        raise ValueError("compute_device must be cpu, gpu, or auto")
    if int(gpu_device_id) < 0:
        raise ValueError("gpu_device_id must be >= 0")

    if requested in {"gpu", "auto"}:
        # Must happen before a worker imports PennyLane/CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_device_id))

    gpu_ok = _gpu_plugin_available()
    cpu_lightning_ok = _cpu_lightning_available()
    if requested == "gpu" and str(qaoa_engine).lower() != "pennylane":
        raise ValueError("--compute-device gpu requires --qaoa-engine pennylane; the NumPy engine is CPU-only")
    if requested == "gpu" and not gpu_ok:
        raise RuntimeError(
            "--compute-device gpu was requested, but PennyLane lightning.gpu was not detected. "
            "Install pennylane-lightning-gpu matching your CUDA and PennyLane versions."
        )
    resolved = "gpu" if (requested == "gpu" or (requested == "auto" and gpu_ok and str(qaoa_engine).lower() == "pennylane")) else "cpu"

    explicit = str(qaoa_device).strip()
    if explicit and explicit.lower() != "auto":
        device = explicit
    else:
        # default.qubit is a Python statevector simulator; OMP_NUM_THREADS does
        # not turn a single tiny QNode into an 8-core workload. On CPU, prefer
        # Lightning-Qubit so --threads-per-worker can control its OpenMP kernels
        # and parallel adjoint-observable work.
        if resolved == "gpu":
            device = "lightning.gpu"
        elif str(qaoa_engine).lower() == "pennylane" and cpu_lightning_ok:
            device = "lightning.qubit"
        else:
            device = "default.qubit"
    return resolved, device, gpu_ok, cpu_lightning_ok


def _validate_pennylane_runtime(device_name: str) -> str:
    """Import PennyLane and instantiate the selected device.

    Return the PennyLane version on success.  Import-time dependency failures
    are deliberately preserved so the CLI reports the real broken package
    instead of incorrectly claiming that PennyLane is not installed.
    """
    try:
        import pennylane as qml
    except Exception as exc:
        raise RuntimeError(
            f"PennyLane is installed but could not be imported: "
            f"{type(exc).__name__}: {exc}. "
            "Re-run python setup.py (CPU) or python setup.py --gpu (NVIDIA GPU) "
            "to install the repository's compatibility-pinned dependencies."
        ) from exc

    try:
        qml.device(str(device_name), wires=1)
    except Exception as exc:
        raise RuntimeError(
            f"PennyLane imported successfully, but device {device_name!r} could not "
            f"be initialized: {type(exc).__name__}: {exc}"
        ) from exc

    return str(getattr(qml, "__version__", "unknown"))


def _worker_count(requested: int, resolved_device: str, task_count: int) -> int:
    requested = int(requested)
    if requested < 0:
        raise ValueError("--max-parallel-instances must be >= 0")
    if task_count <= 0:
        return 1
    if requested == 0:
        if resolved_device == "gpu":
            return 1
        return min(task_count, max(1, min(4, os.cpu_count() or 1)))
    return min(task_count, max(1, requested))


def _set_worker_thread_env(threads_per_worker: int) -> None:
    # Re-apply after argument parsing for clarity. The same values were already
    # installed before heavy imports, which matters for native thread runtimes.
    _apply_native_thread_env(threads_per_worker)


def _ordered_parallel(worker, payloads: Sequence[Any], workers: int, label: str,
                      max_tasks_per_worker: int = 0):
    payloads = list(payloads)
    if workers <= 1:
        for pos, payload in enumerate(payloads, 1):
            print(f"{label} {pos}/{len(payloads)}", flush=True)
            yield worker(payload)
        return

    print(f"{label}: up to {workers} instances concurrently", flush=True)
    ctx = mp.get_context("spawn")
    ready = {}
    next_idx = 0
    # Reuse workers by default: launching a new interpreter for every small
    # exact-enumeration instance adds minutes of unnecessary spawn overhead.
    # Explicit max_tasks_per_worker=1 remains available for memory isolation.
    worker_kwargs = dict(max_workers=workers, mp_context=ctx)
    if max_tasks_per_worker > 0:
        worker_kwargs['max_tasks_per_child'] = max_tasks_per_worker
    with ProcessPoolExecutor(**worker_kwargs) as executor:
        futures = {executor.submit(worker, payload): i for i, payload in enumerate(payloads)}
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            ready[i] = fut.result()
            completed += 1
            print(f"{label} completed {completed}/{len(payloads)}", flush=True)
            while next_idx in ready:
                yield ready.pop(next_idx)
                next_idx += 1



# Exactly one orchestration entry point. Study implementations live in explicit
# modules under QUBO_Coefficient_Calibration/Experiment; this file only orchestrates them.
import hashlib
import math

from QUBO_Coefficient_Calibration.Experiment.optional import (
    optimum_theorem_rows, theory_instance, optional_nx_instance,
    optional_multiplier_instance, DEFAULT_OPTIONAL_POWERS, DEFAULT_OPTIMA_RATIOS,
    CONSTRUCTIONS, paired_construction_rows, optional_candidate_screen,
)

STUDIES = (
    'theorems', 'conservativeness', 'spectral', 'penalty-multiplier',
    'penalty-nx', 'budget-heatmap', 'optional-nx', 'optional-multiplier',
)
TARGET_PDFS = (
    'theorem_band.pdf','theorem_local.pdf','theorem_global.pdf',
    'theorem_ordering.pdf','theorem_optima.pdf','spectral.pdf',
    'conservativeness.pdf','penalty_multiplier.pdf','penalty_nx.pdf',
    'budget_heatmap.pdf','optional_nx.pdf','optional_multiplier.pdf',
)
GENERATED_CSV = (
    'theorem_thresholds.csv','theorem_sweeps.csv','ordering_sweep.csv',
    'theorem_local.csv','theorem_global.csv','theorem_band.csv',
    'theorem_ordering.csv','theorem_optima.csv','theorem_optima_audit.csv',
    'theorem_optima_boundary_checks.csv',
    'conservativeness.csv','spectral_gap.csv','qaoa_penalty.csv',
    'scaling.csv','budget_robustness.csv','optional_nx.csv','optional_nx_paired.csv',
    'optional_nx_audit.csv','optional_multiplier.csv',
    'optional_multiplier_paired.csv','optional_multiplier_audit.csv',
    'optional_seed_screen.csv','seed_ledger.csv',
)


def _seed(master: int, family: str, n: int, idx: int) -> int:
    # Stable across OSes, independent of selected stage/order/process count.
    # Share the 'optional' seed family across the optional Nx and lambda sweep.
    payload = f'{int(master)}::{family}::{int(n)}::{int(idx)}'.encode('ascii')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def _theorem_worker(config):
    i, seed, n, k, optional_scale, ratios = config
    prob = build_random_cardinality_problem(seed=int(seed),n=int(n),k=int(k))
    summary = theorem_summary(prob,optional_scale=optional_scale)
    summary['instance'] = i
    # Retain the original theorem schemas and sweeps exactly.
    generic = theorem_sweep(prob,[-.75,-.5,-.25,-.1,0.,.05,.1,.25,.5,1.,2.],optional_scale)
    order = ordering_sweep(prob,[0.,.25,.5,.9,1.,1.1,1.5,2.])
    for row in generic+order:
        row['instance'] = i
    local,glob,band,ordering = theorem_plot_rows(prob,i,optional_scale=optional_scale)
    audit,optima = optimum_theorem_rows(prob,i,ratios)
    validation, boundary_probes = theory_instance(prob,i)
    if not int(validation["all_checks_passed"]):
        raise AssertionError(f"Optional theorem failed exact checks for instance {i}")
    summary["optional_theorem_all_checks_passed"] = 1
    summary.update({key:audit[key] for key in
                    ('lambda_disp_exact','lambda_opt_cert','lambda_opt_struct')})
    return dict(instance=i,summary=[summary],generic=generic,order=order,
                local=local,global_=glob,band=band,ordering=ordering,
                optima=optima,optima_audit=[audit],optima_boundary_checks=boundary_probes)


def _conserv_worker(config):
    i,seed,n,k,scale=config
    prob=build_random_cardinality_problem(seed=seed,n=n,k=k)
    return {'conservativeness':[conservativeness_row(prob,i,optional_scale=scale)]}


def _spectral_worker(config):
    i,seed,n,k,multipliers=config
    prob=build_random_cardinality_problem(seed=seed,n=n,k=k)
    rows=run_spectral_instance(prob,penalty_multipliers=multipliers)
    for r in rows: r.update(instance=i,instance_index=i)
    return {'spectral_gap':rows}


def _penalty_worker(config):
    i,seed,n,k,common,multipliers=config
    prob=build_random_cardinality_problem(seed=seed,n=n,k=k)
    rows=run_qaoa_penalty_instance(prob,penalty_multipliers=multipliers,**common)
    for r in rows:r.update(instance=i,instance_index=i)
    return {'qaoa_penalty':rows}


def _scaling_worker(config):
    i,seed,n,k,common=config
    prob=build_random_cardinality_problem(seed=seed,n=n,k=k)
    rows=run_qaoa_scaling_instance(prob,**common)
    for r in rows:r.update(instance=i,instance_index=i)
    return {'scaling':rows}


def _budget_worker(config):
    i,seed,n,k,common,budgets,multipliers=config
    prob=build_random_cardinality_problem(seed=seed,n=n,k=k)
    common=dict(common);common.pop('maxiter',None)
    rows=run_qaoa_budget_instance(prob,budgets=budgets,penalty_multipliers=multipliers,**common)
    for r in rows:r.update(instance=i,instance_index=i)
    return {'budget_robustness':rows}


def _optional_nx_worker(config):
    i,seed,n,k,attempt_index,common,fractions,constructions,max_linear=config
    # Use EXACTLY the same generator settings as the QAOA-independent
    # prescreen.  Otherwise a seed could silently select a different problem.
    prob=build_random_cardinality_problem(
        seed=seed,n=n,k=k,
        require_finite_ordering_boundary=False)
    audit,rows=optional_nx_instance(prob,i,**fractions,constructions=constructions,
                                    marginal_max_linear=max_linear,**common)
    if not rows:
        raise AssertionError('A prescreened Nx instance was unexpectedly excluded')
    audit['generation_attempt_index']=attempt_index
    for row in rows:row['generation_attempt_index']=attempt_index
    return {'optional_nx':rows,'optional_nx_audit':[audit],
            'optional_nx_paired':paired_construction_rows(rows)}


def _optional_multiplier_worker(config):
    i,seed,n,k,attempt_index,common,powers,mode,constructions,max_linear=config
    prob=build_random_cardinality_problem(
        seed=seed,n=n,k=k,
        require_finite_ordering_boundary=False)
    audit,rows=optional_multiplier_instance(
        prob,i,powers=powers,mode=mode,
        constructions=constructions,marginal_max_linear=max_linear,**common)
    if not rows:
        raise AssertionError('A prescreened multiplier instance was unexpectedly excluded')
    audit['generation_attempt_index']=attempt_index
    for row in rows:row['generation_attempt_index']=attempt_index
    return {'optional_multiplier':rows,'optional_multiplier_audit':[audit],
            'optional_multiplier_paired':paired_construction_rows(rows,multiplier=True)}


def _parse_csv_numbers(raw, typ=float, *, allow_zero=True):
    try:
        result=[typ(v.strip()) for v in str(raw).split(',') if v.strip()]
    except ValueError as exc:
        raise ValueError(f'invalid comma-separated list: {raw}') from exc
    if not result or len(result)!=len(set(result)) or any(v < 0 if allow_zero else v<=0 for v in result):
        raise ValueError(f'list must be nonempty, distinct, and nonnegative: {raw}')
    return result


def _check_size(value, name):
    if value < 2 or value > 16:
        raise ValueError(f'{name}: n must be 2..16 (exact enumeration/2^n statevector)')


def _record_seeds(master, family, n, count, ledger):
    return [(i, _seed(master,family,n,i),n,n//2) for i in range(count)]


def _select_optional_candidates(master, n, requested, constructions, max_linear,
                                max_seed_attempts):
    """Deterministically select exactly `requested` finite-threshold candidates.

    Rejections are not QAOA runs: generation and exact calibration happen
    before submitting any optional QAOA work.  The attempt index is the input
    to the reproducible SHA-256 seed protocol, not the accepted instance index.
    A configurable cap prevents an infinite search for pathological settings.
    """
    cap = int(max_seed_attempts) if max_seed_attempts else max(500, 100 * requested)
    selected, audit = [], []
    for attempt_index in range(cap):
        seed = _seed(master, 'optional', n, attempt_index)
        problem = build_random_cardinality_problem(
            seed=seed, n=n, k=n//2, require_finite_ordering_boundary=False)
        check = optional_candidate_screen(problem, constructions, max_linear)
        check.update(generation_attempt_index=attempt_index,
                     accepted_instance_index=(len(selected) if check['accepted'] else ''),
                     requested_accepted_instances=requested)
        audit.append(check)
        if check['accepted']:
            selected.append((len(selected), seed, n, n//2, attempt_index))
            if len(selected) == requested:
                return selected, audit
    raise RuntimeError(
        f'Only {len(selected)}/{requested} eligible optional instances at n={n} '
        f'after {cap} attempts. Increase --optional-max-seed-attempts, change '
        'the construction(s), or report the restrictive finite-threshold cohort.'
    )


def _qaoa_kw(args, seed_offset=0, *, restarts=None, maxiter=None):
    return dict(p=args.qaoa_p,restarts=restarts or args.qaoa_restarts,
                maxiter=maxiter or args.qaoa_maxiter, stepsize=args.qaoa_stepsize,
                optimizer=args.qaoa_optimizer,device_name=args.qaoa_device_resolved,
                seed_base=args.master_seed+seed_offset,engine=args.qaoa_engine)


def _run_selected(args, selected, output):
    # Each stage writes its CSVs immediately after each instance; failures
    # therefore leave partial, inspectable data, never a silently claimed run.
    all_rows={}
    ledger=[]
    counts={}
    qaoa_studies={'penalty-multiplier','penalty-nx','budget-heatmap','optional-nx','optional-multiplier'}
    # Prescreen each problem size ONCE and reuse its accepted seed prefix across
    # Nx and multiplier studies.  A request for 30 instances means 30 completed
    # matched QAOA instances, not 30 candidates with silent exclusions.
    optional_counts={}
    if 'optional-nx' in selected:
        for n in args.optional_nx_sizes:
            optional_counts[n]=max(optional_counts.get(n,0),args.optional_nx_instances)
    if 'optional-multiplier' in selected:
        n=args.optional_multiplier_n
        optional_counts[n]=max(optional_counts.get(n,0),args.optional_multiplier_instances)
    optional_cohorts={}
    screen=[]
    for n, target in sorted(optional_counts.items()):
        accepted, attempts=_select_optional_candidates(
            args.master_seed,n,target,args.optional_constructions,
            args.marginal_max_linear,args.optional_max_seed_attempts)
        optional_cohorts[n]=accepted
        screen.extend(attempts)
        print(f'[optional eligibility] n={n}: {len(accepted)} accepted / '
              f'{len(attempts)} attempted; {len(attempts)-len(accepted)} rejected',
              flush=True)
    if screen:
        write_csv(output/'optional_seed_screen.csv',screen)
    for stage in selected:
        payloads=[]
        worker=None
        if stage=='theorems':
            n=int(args.theorem_n);k=args.theorem_k if args.theorem_k is not None else n//2
            for i in range(args.instances):
                seed=_seed(args.master_seed,'theorem',n,i)
                ledger.append(dict(study=stage,instance=i,n=n,k=k,seed=seed))
                payloads.append((i,seed,n,k,args.theorem_optional_scale,args.optima_ratios))
            worker=_theorem_worker
        elif stage=='conservativeness':
            for n in args.conserv_n:
                for i in range(args.conservativeness_instances_per_size):
                    seed=_seed(args.master_seed,'conserv',n,i)
                    ledger.append(dict(study=stage,instance=i,n=n,k=n//2,seed=seed))
                    payloads.append((i,seed,n,n//2,args.theorem_optional_scale))
            worker=_conserv_worker
        elif stage=='spectral':
            n=int(args.penalty_n)
            for i in range(args.penalty_instances):
                seed=_seed(args.master_seed,'penalty',n,i)
                ledger.append(dict(study=stage,instance=i,n=n,k=n//2,seed=seed))
                payloads.append((i,seed,n,n//2,args.spectral_multipliers))
            worker=_spectral_worker
        elif stage=='penalty-multiplier':
            n=int(args.penalty_n)
            for i in range(args.penalty_instances):
                seed=_seed(args.master_seed,'penalty',n,i)
                ledger.append(dict(study=stage,instance=i,n=n,k=n//2,seed=seed))
                payloads.append((i,seed,n,n//2,_qaoa_kw(args,101),args.penalty_multipliers))
            worker=_penalty_worker
        elif stage=='penalty-nx':
            for n in args.penalty_nx_sizes:
                for i in range(args.penalty_nx_instances):
                    seed=_seed(args.master_seed,'penalty-nx',n,i)
                    ledger.append(dict(study=stage,instance=i,n=n,k=n//2,seed=seed))
                    payloads.append((i,seed,n,n//2,
                        _qaoa_kw(args,201,restarts=args.nx_restarts,maxiter=args.nx_maxiter)))
            worker=_scaling_worker
        elif stage=='budget-heatmap':
            n=int(args.budget_n)
            for i in range(args.budget_instances):
                seed=_seed(args.master_seed,'budget',n,i)
                ledger.append(dict(study=stage,instance=i,n=n,k=n//2,seed=seed))
                payloads.append((i,seed,n,n//2,_qaoa_kw(args,301),
                                 args.budget_iterations,args.budget_multipliers))
            worker=_budget_worker
        elif stage=='optional-nx':
            fractions=dict(exact_fraction=args.optional_order_fraction,
                           surrogate_fraction=args.optional_surrogate_fraction,
                           optimum_fraction=args.optional_disp_fraction,
                           above_fraction=args.optional_above_fraction)
            common=dict(_qaoa_kw(args,401,restarts=args.optional_nx_restarts,
                                 maxiter=args.optional_nx_maxiter),
                        penalty_basis=args.optional_penalty_basis,
                        penalty_factor=args.optional_penalty_factor,
                        shots=args.optional_shots)
            for n in args.optional_nx_sizes:
                for i,seed,_,k,attempt_index in optional_cohorts[n][:args.optional_nx_instances]:
                    ledger.append(dict(study=stage,instance=i,n=n,k=k,seed=seed,
                                       generation_attempt_index=attempt_index))
                    payloads.append((i,seed,n,k,attempt_index,common,fractions,
                                     args.optional_constructions,args.marginal_max_linear))
            worker=_optional_nx_worker
        elif stage=='optional-multiplier':
            n=int(args.optional_multiplier_n)
            common=dict(_qaoa_kw(args,401,restarts=args.optional_multiplier_restarts,
                                 maxiter=args.optional_multiplier_maxiter),
                        penalty_basis=args.optional_penalty_basis,
                        penalty_factor=args.optional_penalty_factor,
                        shots=args.optional_shots)
            for i,seed,_,k,attempt_index in optional_cohorts[n][:args.optional_multiplier_instances]:
                ledger.append(dict(study=stage,instance=i,n=n,k=k,seed=seed,
                                   generation_attempt_index=attempt_index))
                payloads.append((i,seed,n,k,attempt_index,common,args.optional_powers,
                                 args.optional_multiplier_mode,
                                 args.optional_constructions,args.marginal_max_linear))
            worker=_optional_multiplier_worker
        else:
            raise AssertionError(stage)
        device='gpu' if stage in qaoa_studies and args.compute_device_resolved=='gpu' else 'cpu'
        workers=_worker_count(args.max_parallel_instances,device,len(payloads))
        stage_data={}
        for item in _ordered_parallel(worker,payloads,workers,f'[{stage}]',
                                      args.max_tasks_per_worker):
            # The original theorem worker keeps old compatibility keys.
            mapping={
                'summary':'theorem_thresholds','generic':'theorem_sweeps',
                'order':'ordering_sweep','local':'theorem_local',
                'global_':'theorem_global','band':'theorem_band',
                'ordering':'theorem_ordering','optima':'theorem_optima',
                'optima_audit':'theorem_optima_audit',
                'optima_boundary_checks':'theorem_optima_boundary_checks',
                'conservativeness':'conservativeness',
            }
            for name,rows in item.items():
                if name=='instance': continue
                csv_name=mapping.get(name,name)
                stage_data.setdefault(csv_name,[]).extend(rows)
                write_csv(output/(csv_name+'.csv'),stage_data[csv_name])
        counts[stage]={key:len(val) for key,val in stage_data.items()}
        all_rows.update(stage_data)
        write_csv(output/'seed_ledger.csv',ledger)
        print(f'[{stage}] {len(payloads)} instances complete; rows: {counts[stage]}',flush=True)
    return counts


def _parser():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group=ap.add_argument_group('Stage selection and reproducibility')
    group.add_argument('--out',default='results')
    group.add_argument('--master-seed',type=int,default=20260929)
    group.add_argument('--studies',default='',help='Comma-separated studies; if omitted: theorems,conservativeness')
    group.add_argument('--run-study',action='append',default=[],choices=STUDIES,
                       help='Repeatable individual study selector (combined with --studies)')
    group.add_argument('--exclude-studies',default='',help='Comma-separated study names to skip')
    group.add_argument('--all',action='store_true',help='Enable all eight stages / twelve paper PDFs')
    group.add_argument('--list-studies',action='store_true')
    group.add_argument('--plot-only',action='store_true',help='Regenerate available figures from saved CSVs')
    group.add_argument('--no-plot',action='store_true')
    group.add_argument('--force',action='store_true',help='Remove ONLY known generated result files before a NEW run')
    group.add_argument('--dry-run',action='store_true',help='List stages and work estimates without executing')

    group=ap.add_argument_group('Exact theorem validation')
    group.add_argument('--instances',type=int,default=30)
    group.add_argument('--theorem-n',type=int,default=8)
    group.add_argument('--theorem-k',type=int,default=None)
    group.add_argument('--theorem-optional-scale',type=float,default=.15,
                       help='Optional scale in penalty theorems; optimum theorem always calibrates its own lambda')
    group.add_argument('--optima-ratios',default=','.join(str(v) for v in DEFAULT_OPTIMA_RATIOS),
                       help='Nonnegative lambda/lambda_disp probes, MUST include 1 for tie')

    group=ap.add_argument_group('Conservativeness (five panels)')
    group.add_argument('--conservativeness-n',default='6,8,10,12')
    group.add_argument('--conservativeness-instances-per-size',type=int,default=30)

    group=ap.add_argument_group('Shared QAOA and parallel settings')
    group.add_argument('--qaoa-engine', choices=('pennylane',), default='pennylane',
                       help='QAOA engine (repository is trimmed to PennyLane only)')
    group.add_argument('--qaoa-p',type=int,default=3)
    group.add_argument('--qaoa-restarts',type=int,default=3)
    group.add_argument('--qaoa-maxiter',type=int,default=100)
    group.add_argument('--qaoa-stepsize',type=float,default=.05)
    group.add_argument('--qaoa-optimizer',choices=('adam','gradient-descent'),default='adam')
    group.add_argument('--compute-device',choices=('cpu','gpu','auto'),default='cpu')
    group.add_argument('--qaoa-device',default='auto')
    group.add_argument('--gpu-device-id',type=int,default=0)
    group.add_argument('--max-parallel-instances',type=int,default=0,
                       help='0: auto (up to 4 CPU instances / one GPU instance)')
    group.add_argument('--threads-per-worker',type=int,default=1)
    group.add_argument('--max-tasks-per-worker',type=int,default=0,
                       help='0 reuses worker processes; 1 restarts after every instance for memory isolation')

    group=ap.add_argument_group('Spectral and penalty multiplier')
    group.add_argument('--penalty-n',type=int,default=8)
    group.add_argument('--penalty-instances',type=int,default=30)
    group.add_argument('--penalty-multipliers',default=','.join(map(str,QAOA_P_OVER_GLOBAL)))
    group.add_argument('--spectral-multipliers',default=','.join(map(str,SPECTRAL_P_OVER_GLOBAL)))

    group=ap.add_argument_group('Penalty scaling versus Nx')
    group.add_argument('--penalty-nx-sizes',default='6,8,10,12')
    group.add_argument('--penalty-nx-instances',type=int,default=30)
    group.add_argument('--nx-restarts',type=int,default=3)
    group.add_argument('--nx-maxiter',type=int,default=100)

    group=ap.add_argument_group('Penalty iteration-budget heatmap')
    group.add_argument('--budget-n',type=int,default=8)
    group.add_argument('--budget-instances',type=int,default=30)
    group.add_argument('--budget-iterations',default=','.join(map(str,DEFAULT_WORK_BUDGETS)))
    group.add_argument('--budget-multipliers',default=','.join(map(str,BUDGET_P_OVER_GLOBAL)))

    group=ap.add_argument_group('Optional QAOA construction comparison')
    group.add_argument('--optional-constructions',default='random,marginal',
                       help='random, marginal, or random,marginal; exact comparison requires matching eligible instances')
    group.add_argument('--marginal-max-linear',type=float,default=2.,
                       help='Max absolute centered marginal-cost linear coefficient (coefficient-only normalization)')
    group.add_argument('--optional-max-seed-attempts',type=int,default=0,
                       help='Candidate-seed cap PER Nx size; 0 means max(500,100*requested eligible instances). '
                            'Rejected infinite-bound seeds are logged, never sent to QAOA.')

    group=ap.add_argument_group('Optional Nx (baseline + 2 constructions x 5 coefficient policies)')
    group.add_argument('--optional-nx-sizes',default='6,8,10,12')
    group.add_argument('--optional-nx-instances',type=int,default=30)
    group.add_argument('--optional-nx-restarts',type=int,default=3)
    group.add_argument('--optional-nx-maxiter',type=int,default=100)
    group.add_argument('--optional-order-fraction',type=float,default=.95)
    group.add_argument('--optional-surrogate-fraction',type=float,default=.95)
    group.add_argument('--optional-disp-fraction',type=float,default=.95)
    group.add_argument('--optional-above-fraction',type=float,default=1.05)

    group=ap.add_argument_group('Optional coefficient multiplier')
    group.add_argument('--optional-multiplier-mode',choices=('ordering-tight','common'),
                       default='ordering-tight',
                       help='One power-of-two sweep for each construction, relative to its own sharp ordering threshold; common is a legacy alias')
    group.add_argument('--optional-multiplier-n',type=int,default=8)
    group.add_argument('--optional-multiplier-instances',type=int,default=30)
    group.add_argument('--optional-powers',default=','.join(map(str,DEFAULT_OPTIONAL_POWERS)),
                       help='Powers e for lambda=2**e times the construction-specific exact ordering threshold')
    group.add_argument('--optional-multiplier-restarts',type=int,default=3)
    group.add_argument('--optional-multiplier-maxiter',type=int,default=100)
    group.add_argument('--optional-shots',type=int,default=32,
                       help='Iid measurements for expected best-feasible returned solution')
    group.add_argument('--optional-penalty-basis',choices=('exact-global','certified-global'),
                       default='exact-global',help='One constant P per instance across all optional coefficient settings')
    group.add_argument('--optional-penalty-factor',type=float,default=1.05)
    return ap


def _validate(args,ap,selected):
    if args.master_seed<0 or args.master_seed>2**32-1:
        ap.error('master seed must be in [0, 2**32-1]')
    for key in ('instances','conservativeness_instances_per_size','penalty_instances',
                'penalty_nx_instances','budget_instances','optional_nx_instances',
                'optional_multiplier_instances'):
        if getattr(args,key)<1: ap.error(f'--{key.replace("_","-")} must be >=1')
    for n in (args.theorem_n,args.penalty_n,args.budget_n,args.optional_multiplier_n,
              *args.conserv_n,*args.penalty_nx_sizes,*args.optional_nx_sizes):
        try:_check_size(n,'problem size')
        except ValueError as e:ap.error(str(e))
    if args.theorem_k is not None and not 0<args.theorem_k<args.theorem_n:
        ap.error('theorem k must satisfy 0<k<n')
    if args.theorem_optional_scale<0: ap.error('theorem optional scale must be >=0')
    for key in ('qaoa_p','qaoa_restarts','qaoa_maxiter','nx_restarts','nx_maxiter',
                'optional_nx_restarts','optional_nx_maxiter','optional_multiplier_restarts',
                'optional_multiplier_maxiter','optional_shots','threads_per_worker'):
        if getattr(args,key)<1:ap.error(f'--{key.replace("_","-")} must be >=1')
    if not 0<args.qaoa_stepsize or args.max_parallel_instances<0 or args.max_tasks_per_worker<0:
        ap.error('stepsize must be >0, parallel workers >=0')
    if args.optional_penalty_factor<=1:ap.error('optional penalty factor must exceed 1')
    if not math.isfinite(args.marginal_max_linear) or args.marginal_max_linear<=0:
        ap.error('--marginal-max-linear must be finite and positive')
    if args.optional_max_seed_attempts<0:
        ap.error('--optional-max-seed-attempts must be >=0 (0: automatic cap)')
    if any(not (0<x<1) for x in (args.optional_order_fraction,args.optional_surrogate_fraction,args.optional_disp_fraction)):
        ap.error('safe optional Nx coefficient fractions must lie strictly within (0,1)')
    if args.optional_above_fraction<=1:ap.error('above-displacement factor must exceed 1')
    if 1. not in args.optima_ratios:ap.error('--optima-ratios must contain 1 (the tie boundary)')
    if not args.optional_powers or max(args.optional_powers)>10 or min(args.optional_powers)<-16:
        ap.error('optional powers must be within [-16,10]')


def _work_estimates(args,selected):
    return {
        'theorems':args.instances,
        'conservativeness':len(args.conserv_n)*args.conservativeness_instances_per_size,
        'spectral':args.penalty_instances,
        'penalty-multiplier':args.penalty_instances*len(args.penalty_multipliers)*args.qaoa_restarts,
        'penalty-nx':len(args.penalty_nx_sizes)*args.penalty_nx_instances*7*args.nx_restarts,
        'budget-heatmap':args.budget_instances*len(args.budget_multipliers)*args.qaoa_restarts,
        'optional-nx':len(args.optional_nx_sizes)*args.optional_nx_instances*(
            1+5*len(args.optional_constructions))*args.optional_nx_restarts,
        'optional-multiplier':args.optional_multiplier_instances*(
            1+len(args.optional_constructions)*len(args.optional_powers)
        )*args.optional_multiplier_restarts,
    }


def main():
    ap=_parser()
    args=ap.parse_args()
    if args.list_studies:
        print('\n'.join(STUDIES))
        return
    out=Path(args.out)
    if args.plot_only:
        if args.no_plot or args.all or args.studies or args.run_study:
            ap.error('--plot-only cannot be combined with run/plot suppression flags')
        if not out.exists():ap.error(f'plot-only output folder not found: {out}')
        import plot
        plot.generate_plots(out)
        print('Replotted available experiment CSVs:', out)
        return
    try:
        args.conserv_n=_parse_csv_numbers(args.conservativeness_n,int,allow_zero=False)
        args.penalty_nx_sizes=_parse_csv_numbers(args.penalty_nx_sizes,int,allow_zero=False)
        args.optional_nx_sizes=_parse_csv_numbers(args.optional_nx_sizes,int,allow_zero=False)
        args.optional_constructions=tuple(x.strip().lower() for x in
                                          args.optional_constructions.split(',') if x.strip())
        if (not args.optional_constructions or
                len(args.optional_constructions) != len(set(args.optional_constructions)) or
                any(x not in CONSTRUCTIONS for x in args.optional_constructions)):
            raise ValueError('--optional-constructions must be random, marginal, or random,marginal without duplicates')
        args.optima_ratios=_parse_csv_numbers(args.optima_ratios,float)
        args.penalty_multipliers=_parse_csv_numbers(args.penalty_multipliers,float,allow_zero=False)
        args.spectral_multipliers=_parse_csv_numbers(args.spectral_multipliers,float,allow_zero=False)
        args.budget_iterations=_parse_csv_numbers(args.budget_iterations,int,allow_zero=False)
        args.budget_multipliers=_parse_csv_numbers(args.budget_multipliers,float,allow_zero=False)
        powers=[int(part.strip()) for part in str(args.optional_powers).split(',') if part.strip()]
        if len(powers)!=len(set(powers)):raise ValueError('duplicate optional powers')
        args.optional_powers=sorted(powers)
    except ValueError as e:ap.error(str(e))
    names=list(STUDIES) if args.all else (
        ['theorems','conservativeness'] if not args.studies and not args.run_study
        else [x.strip() for x in args.studies.split(',') if x.strip()]+args.run_study)
    exclude={x.strip() for x in args.exclude_studies.split(',') if x.strip()}
    if any(x not in STUDIES for x in names+list(exclude)):
        ap.error('unknown study: available: '+', '.join(STUDIES))
    selected=[x for x in STUDIES if x in names and x not in exclude]
    if not selected:ap.error('no studies selected')
    _validate(args,ap,selected)
    print('Selected studies: '+', '.join(selected))
    est=_work_estimates(args,selected)
    for name in selected:
        qualifier='instance enumerations' if name in ('theorems','conservativeness','spectral') else 'QAOA restart trajectories'
        print(f'  {name}: ~{est[name]} {qualifier}')
    if args.dry_run:return
    if out.exists():
        existing=[out/name for name in (*GENERATED_CSV,*TARGET_PDFS,'experiment_manifest.json') if (out/name).exists()]
        if existing and not args.force:
            ap.error('output already contains generated results; choose new --out or pass --force to delete known generated files')
        if args.force:
            for path in existing: path.unlink()
    out.mkdir(parents=True,exist_ok=True)
    _apply_native_thread_env(args.threads_per_worker)
    needs_qaoa=any(x in selected for x in ('penalty-multiplier','penalty-nx','budget-heatmap','optional-nx','optional-multiplier'))
    if needs_qaoa:
        resolved,device,gpu_found,cpu_found=_resolve_device(
            args.compute_device,args.qaoa_device,args.qaoa_engine,args.gpu_device_id)
        if args.qaoa_engine=='pennylane':
            try:
                pennylane_version = _validate_pennylane_runtime(device)
            except RuntimeError as exc:
                ap.error(str(exc))
            print(f'PennyLane runtime: {pennylane_version}')
        print(f'QAOA backend: {args.qaoa_engine} / {device}; compute={resolved}')
    else:
        resolved='cpu';device='unused';gpu_found=cpu_found=False
    args.compute_device_resolved=resolved
    args.qaoa_device_resolved=device
    counts=_run_selected(args,selected,out)
    manifest={
        'project':'unified fixed-cardinality QUBO; primary objective alone evaluates all returned solutions',
        'master_seed':args.master_seed,'selected_studies':selected,'counts':counts,
        'qaoa_engine':args.qaoa_engine,'qaoa_device':device,
        'plot_style_requested':'SciencePlots: science + no-latex',
        'scienceplots_installed':importlib.util.find_spec('scienceplots') is not None,
        'notes':[
            'Each study has reproducible problem seeds. Optional Nx and multiplier use the SAME first accepted seed prefix for equal n, master seed, and selected constructions, even if their target counts differ.',
            'Each QAOA configuration uses a matched set of restart initialization seeds; multi-start is selected by perturbed variational energy, not oracle primary outcomes.',
            'Optional QAOA reports single-shot primary-optimum probability, and analytic expected best-of-S feasible primary objective conditional on observing a feasible shot.',
            'The optional penalty stays fixed across every tested lambda for an instance; above-displacement is a deliberate violation of original primary-optimum preservation.',
            'Optional QAOA compares random and non-oracular marginal-cost templates on matched eligible primary instances; baseline, P and restart seeds are common.',
            'Marginal guidance is centered and normalized a_i + (k-1)/(n-1) sum_{j != i} b_ij; it never uses exact optima, feasible enumeration, or QAOA performance.',
            'The optional runner screens candidates before QAOA and keeps trying successive SHA-256 seeds until EXACTLY the requested number of matched finite-bound instances is selected (or the explicit safety cap fails).',
            'All attempted optional candidate seeds, rejected cases, exact thresholds and reasons appear in optional_seed_screen.csv; this conditional cohort is NOT representative of unfiltered generated problems.',
            'Optional multiplier has exactly one sweep per construction: lambda=2**power times THAT construction\'s exact ordering threshold; other bounds are plot annotations, not extra QAOA series.',
            'At lambda=exact displacement a suboptimal tie invalidates the all-minimizers guarantee even if zero states strictly beat the original primary-optimal set.',
            'Error bars show between-instance SEM, not independent shots.',
        ],
        'settings':{key:value for key,value in vars(args).items() if key not in
                    ('optima_ratios','conserv_n','penalty_nx_sizes','optional_nx_sizes',
                     'penalty_multipliers','spectral_multipliers','budget_iterations',
                     'budget_multipliers','optional_powers','optional_constructions')},
        'lists':{key:getattr(args,key) for key in
                 ('optima_ratios','conserv_n','penalty_nx_sizes','optional_nx_sizes',
                  'penalty_multipliers','spectral_multipliers','budget_iterations',
                  'budget_multipliers','optional_powers','optional_constructions')},
        'python_version':platform.python_version(),
    }
    (out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    if not args.no_plot:
        import plot
        plot.generate_plots(out)
        actual={f.name for f in out.glob('*.pdf')}
        expected={
            'theorems':{'theorem_local.pdf','theorem_global.pdf','theorem_band.pdf','theorem_ordering.pdf','theorem_optima.pdf'},
            'conservativeness':{'conservativeness.pdf'},
            'spectral':{'spectral.pdf'},
            'penalty-multiplier':{'penalty_multiplier.pdf'},
            'penalty-nx':{'penalty_nx.pdf'},
            'budget-heatmap':{'budget_heatmap.pdf'},
            'optional-nx':{'optional_nx.pdf'},
            'optional-multiplier':{'optional_multiplier.pdf'},
        }
        needed=set().union(*(expected[x] for x in selected))
        missing=needed-actual
        if missing:
            print('WARNING: expected PDFs not emitted (see empty/filtered family data): '+', '.join(sorted(missing)),flush=True)
        print('Figure outputs: '+', '.join(sorted(actual)))
    print('Finished:',out,flush=True)


if __name__=='__main__':
    main()
