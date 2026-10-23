#!/usr/bin/env python3
"""Reproduce the paper's production experiment run with frozen parameters.

This wrapper intentionally accepts no experiment overrides.  It launches
``run_experiments.py`` with the exact production arguments used for the paper.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PAPER_ARGS = [
    "--all",
    "--out", "results",
    "--master-seed", "1",

    "--instances", "30",
    "--theorem-n", "8",

    "--conservativeness-n", "6,8,10,12,14,16",
    "--conservativeness-instances-per-size", "30",

    "--penalty-n", "6",
    "--penalty-instances", "30",

    "--penalty-nx-sizes", "6,8,10,12,14,16",
    "--penalty-nx-instances", "30",
    "--nx-restarts", "3",
    "--nx-maxiter", "100",

    "--budget-n", "6",
    "--budget-instances", "30",
    "--budget-iterations", "10,30,50,70,90",

    "--optional-constructions", "random,marginal",
    "--optional-nx-sizes", "6,8,10,12",
    "--optional-nx-instances", "30",
    "--optional-nx-restarts", "3",
    "--optional-nx-maxiter", "100",

    "--optional-multiplier-n", "8",
    "--optional-multiplier-instances", "30",
    "--optional-multiplier-mode", "ordering-tight",
    "--optional-powers=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10",
    "--optional-multiplier-restarts", "3",
    "--optional-multiplier-maxiter", "100",
    "--optional-max-seed-attempts", "0",

    "--qaoa-engine", "pennylane",
    "--qaoa-p", "3",
    "--qaoa-restarts", "3",
    "--qaoa-maxiter", "100",
    "--qaoa-optimizer", "adam",
    "--qaoa-stepsize", "0.05",

    "--compute-device", "cpu",
    "--max-parallel-instances", "4",
    "--threads-per-worker", "7",
]


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit(
            "reproduce_paper_results.py takes no arguments; its settings are "
            "intentionally frozen to the paper-production configuration."
        )

    runner = Path(__file__).resolve().with_name("run_experiments.py")
    subprocess.run([sys.executable, str(runner), *PAPER_ARGS], check=True)


if __name__ == "__main__":
    main()
