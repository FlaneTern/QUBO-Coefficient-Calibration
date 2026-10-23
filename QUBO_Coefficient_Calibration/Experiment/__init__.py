"""Experiment implementations for the fixed-cardinality QUBO study.

Study modules
-------------
- theorem.py: theorem-threshold and ordering studies
- conservativeness.py: certificate-conservativeness study
- spectral.py: spectral-gap study
- penalty.py: penalty-multiplier and penalty-Nx studies
- budget.py: paired optimizer-budget study
- optional.py: optional-term theory, Nx, and multiplier studies

Shared implementation details live in analysis.py and qaoa.py.  The public
orchestration entry point remains the repository-level run_experiments.py.
"""
