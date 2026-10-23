# QUBO Coefficient Calibration

Source code for the paper currently titled **“Calibrating QUBO Coefficients for Feasibility and Ordering Preservation.”**

The repository contains the fixed-cardinality problem definition, exact threshold/certificate analysis, PennyLane QAOA implementation, experiment orchestration, and plotting code required to reproduce the manuscript data and figures.

**To reproduce the paper:** If you want the raw results and figures, [download precomputed results and figures in releases](https://github.com/FlaneTern/QUBO-Coefficient-Calibration/releases). If you want to run the experiments yourself, read **Requirements** and **Installation** first, then run **`python reproduce_paper_results.py`**. This is the frozen paper-reproduction entry point; the other scripts are for additional, customizable experiments and supporting tasks.

> **Research status:** The manuscript and repository interface are still being finalized. File names and small implementation details may change while the paper is in preparation.

> **Reproduction warning:** The frozen paper-production configuration uses the CPU implementation. The GPU environment is provided as an alternative execution path, but it is not the configuration used by `reproduce_paper_results.py` and should not be assumed to reproduce the production run identically until separately validated.


## Precomputed Paper Results

The complete experimental results and manuscript figures
generated using `reproduce_paper_results.py` are available
as downloadable assets in GitHub Releases.

**[Download precomputed results and figures](https://github.com/FlaneTern/QUBO-Coefficient-Calibration/releases)**

The archive contains:
- CSV files with experiment results and supporting data.
- PDF figures generated from the experimental results.
- `experiment_manifest.json` documenting the run configuration.
- `seed_ledger.csv` documenting the generated problem seeds.

All results were generated using the frozen paper-production
configuration with master seed 1 and the CPU backend.

The generated data and figures are distributed separately
to keep the Git repository lightweight.


  

## Requirements

Before attempting the paper-reproduction run, make sure your system meets these requirements:

- **Python 3.11+**
- A standard Python `venv` environment
- Sufficient CPU time and memory for exact enumeration and statevector QAOA experiments up to `n = 16`
- PennyLane with `lightning.qubit` for CPU execution
- Linux, an NVIDIA GPU, a compatible NVIDIA driver, and CUDA 12 runtime packages for the optional `lightning.gpu` environment

Two complete dependency files are provided:

- `requirements-cpu.txt` for the CPU implementation
- `requirements-gpu.txt` for the NVIDIA/CUDA 12 GPU implementation

The GPU requirements file is self-contained and includes the PennyLane Lightning GPU plugin together with the CUDA/cuStateVec runtime packages used by that implementation.

## Installation — CPU setup for paper reproduction

**Use the CPU environment for reproducing the paper.** Install the dependencies before running any experiment script.

Clone the repository and enter its root directory:

```bash
git clone https://github.com/FlaneTern/QUBO-Coefficient-Calibration.git
cd QUBO-Coefficient-Calibration
```

### CPU environment

Create the virtual environment and install the CPU dependencies:

```bash
python setup.py
```

Activate it with:

```bash
source ./venv/bin/activate
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

The CPU setup installs `requirements-cpu.txt` and is the environment used for the frozen paper-production configuration.

## Reproducing the paper — run `reproduce_paper_results.py`

> **IMPORTANT — THIS IS THE FILE TO RUN TO REPRODUCE THE PAPER:** **`reproduce_paper_results.py`** is the dedicated entry point for the paper's frozen production configuration. **Do not substitute `run_experiments.py` (even with `--all`) or `plot.py` for the paper-reproduction command.** Those scripts are for configurable experiments, plotting, or supporting the reproduction workflow.

**After completing the CPU installation above**, from the repository root with the virtual environment activated, run:

```bash
python reproduce_paper_results.py
```

**No command-line arguments are needed or accepted.** The script calls `run_experiments.py` internally with the fixed paper-production parameters (including the CPU backend and master seed) and writes the generated results to:

```text
results/
```

The full run is computationally demanding. Start with a clean `results/` directory: the experiment runner protects existing generated results rather than overwriting them automatically.

The paper-reproduction workflow runs all eight study stages:

1. Exact theorem validation.
2. Conservativeness of analytical certificates across problem sizes.
3. Spectral behavior across penalty multipliers.
4. QAOA performance across penalty multipliers.
5. Penalty behavior across problem sizes.
6. QAOA iteration-budget robustness.
7. Optional-term constructions and coefficient policies across problem sizes.
8. Optional-term multiplier sweeps relative to exact ordering thresholds.

After the studies finish, the runner automatically generates the available manuscript figures. Figures are written as PDF files at 300 DPI alongside the CSV data in `results/`.

For reference, the complete fixed arguments are documented under [Paper production configuration](#paper-production-configuration-reference-only). **For actual paper reproduction, always invoke `python reproduce_paper_results.py` rather than copying and editing those arguments.**

## Paper experiment design and instance generation

The instance counts, problem sizes, and sampling rules below describe the **frozen paper-production run in `reproduce_paper_results.py`**, not the adjustable defaults of `run_experiments.py`. The Python scripts, rather than historical manuscript comments, are the source of truth for these settings.

### Instance counts and cardinality targets

**Every paper study uses 30 independently seeded problem instances per tested problem size**, with multiple parameter settings evaluated on the *same* instances within a study. All tested sizes are even, so the cardinality target is

$$
k = \lfloor n/2 \rfloor = n/2.
$$

| Paper study (`run_experiments.py` stage) | Problem sizes `n` | Target `k` | Instances |
|---|---|---|---|
| Exact theorem and threshold validation (`theorems`) | `8` | `4` | 30 |
| Certificate conservativeness (`conservativeness`) | `6, 8, 10, 12, 14, 16` | `n/2` | 30 per size (180 total) |
| Exact spectral analysis (`spectral`) | `6` | `3` | 30 |
| QAOA penalty-multiplier sweep (`penalty-multiplier`) | `6` | `3` | 30 |
| QAOA penalty-versus-size study (`penalty-nx`) | `6, 8, 10, 12, 14, 16` | `n/2` | 30 per size (180 total) |
| QAOA optimizer-budget robustness (`budget-heatmap`) | `6` | `3` | 30 |
| Optional-term construction/coefficient study (`optional-nx`) | `6, 8, 10, 12` | `n/2` | 30 accepted instances per size (120 total) |
| Optional-coefficient multiplier sweep (`optional-multiplier`) | `8` | `4` | 30 accepted instances |

The optimizer-budget study compares iteration checkpoints **`10, 30, 50, 70, 90`** on paired optimization trajectories. The optional-coefficient multiplier sweep compares powers **`-5` through `10`** (inclusive) relative to each construction's exact ordering threshold. Both settings are fixed by `reproduce_paper_results.py`.

**“Independent” refers to the generation of different problem instances within a size/study cohort, not to every plotted point.** Spectral and penalty-multiplier studies intentionally reuse the same 30 problem seeds at `n=6`; the optional-term size study and optional multiplier study reuse the same accepted 30-instance cohort at `n=8`. Sweeping penalties, coefficients, budgets, or optimizer restarts does not create additional independent problem instances. These matched comparisons are deliberate.

### Random problem construction and structural filters

The shared generator, `build_random_cardinality_problem()` in `QUBO_Coefficient_Calibration/Problem/cardinality.py`, draws small fixed-cardinality binary quadratic problems. For its paper-run defaults:

- **Primary objective:** each linear coefficient is sampled uniformly from the integers `-4` through `4`; each possible quadratic pair is included with probability `0.45` and receives a uniformly sampled nonzero integer coefficient from `-3` through `3`.
- **Random optional template:** linear coefficients are sampled uniformly from the integers `-2` through `2`; each quadratic pair is included with probability `0.20` and assigned `-1` or `+1` with equal probability. The generator ensures the primary linear and optional templates are not identically zero.
- **Cardinality constraint:** feasible bitstrings satisfy `sum(x) = k`, enforced by the penalty `(sum(x) - k)^2` in the QUBO.

Instance filtering is based only on exact structural properties, **never on QAOA performance**:

1. **Nontrivial feasibility penalty (all study families):** reject a draw if *any* unconstrained global minimizer of the primary objective already satisfies `sum(x) = k`. This keeps instances for which enforcing cardinality is genuinely necessary.
2. **Finite, positive random-template ordering boundary (theorem, conservativeness, spectral, penalty-multiplier, penalty-nx, and budget-heatmap):** also reject a draw when its exact ordering-failure threshold is infinite, zero, or otherwise nonpositive/nonfinite. These stages use the generator's default `require_finite_ordering_boundary=True`.

For each derived problem seed, the generator advances its own NumPy pseudorandom stream and redraws until its applicable conditions are satisfied. **The internal limit is 500 draws per problem seed** (`max_attempts=500`); exhaustion raises an error rather than silently substituting an unchecked instance. These filters are part of the reproducible sampled problem distribution, not a post-hoc selection based on algorithm results.

### Additional eligibility screening for optional-term studies

The two optional-term QAOA stages (`optional-nx` and `optional-multiplier`) **do not apply the generator's second, random-template ordering-boundary filter**: they call it with `require_finite_ordering_boundary=False` while retaining the nontrivial-feasibility filter. Instead, they perform a **separate exact, QAOA-independent screening** on the two optional constructions used in the paper:

- `random`: the optional template sampled by the shared generator.
- `marginal`: a coefficient-derived marginal-cost *linear* template constructed from the same primary objective, without using QAOA outcomes or an exact optimizer to construct that template.

A candidate problem is accepted **only if both constructions** have finite, strictly positive values for *all four* calibration quantities needed by the coefficient policies: the exact ordering threshold (`lambda_order_exact`), practical ordering surrogate (`lambda_order_practical`), exact primary-optimum displacement threshold (`lambda_disp_exact`), and structural optimum surrogate (`lambda_opt_struct`). A failure for either construction excludes the candidate from **both** arms of the matched comparison; the program never treats a genuinely infinite threshold as an arbitrary large finite coefficient.

The optional studies continue through deterministically derived **candidate seeds** until they have **30 accepted primary problem instances per requested size**, rather than interpreting 30 attempted seeds as 30 usable experiments. In the paper configuration `--optional-max-seed-attempts 0` selects the automatic **outer cap `max(500, 100 × requested_instances)`**, which is **3,000 candidate seeds when 30 accepted instances are requested**. This outer screening cap is distinct from the generator's **500 internal redraws per candidate seed**. Screening outcomes and rejection reasons are written to `results/optional_seed_screen.csv` before optional QAOA execution.

### Deterministic seeds and audit trail

The paper wrapper fixes `--master-seed 1`. `run_experiments.py` derives problem seeds reproducibly from the master seed, study-family identifier, problem size, and instance (or optional candidate-attempt) index using SHA-256; results do not depend on worker scheduling or study order. `results/seed_ledger.csv` records the selected problem seeds, and optional-stage rows also include the candidate generation-attempt index. `results/experiment_manifest.json` records the run configuration. Together with the generator and screening code, these files document which instances entered the reported results.

## Additional custom experiments — use `run_experiments.py`

**This section is for experiments beyond the frozen paper-production run.** To reproduce the paper, use **`python reproduce_paper_results.py`** as shown above.

For different study subsets, parameters, output locations, and optional GPU execution, use the configurable entry point **`run_experiments.py`**.

Display all available options with:

```bash
python run_experiments.py --help
```

List the eight study names with:

```bash
python run_experiments.py --list-studies
```

If no study selector is supplied, the runner executes only:

```text
theorems, conservativeness
```

**Custom all-studies run (not the frozen paper configuration):**

```bash
python run_experiments.py --all --out custom_results
```

**Important:** `--all` selects the eight studies but does **not** apply every fixed paper-production parameter. Keep custom outputs in a separate directory (such as `custom_results/`) to avoid mixing them with the paper reproduction.

Before starting an expensive run, the selected stages and approximate work counts can be checked without executing them:

```bash
python run_experiments.py --all --out custom_results --dry-run
```

### Running selected experiments

Studies can be selected individually with the repeatable `--run-study` option:

```bash
python run_experiments.py \
  --run-study theorems \
  --run-study spectral \
  --out selected_results
```

They can also be supplied as a comma-separated list:

```bash
python run_experiments.py \
  --studies theorems,spectral,penalty-multiplier \
  --out selected_results
```

Available study names are:

| Study | Purpose | Main figure output |
|---|---|---|
| `theorems` | Exact validation of local, global, band, ordering, and primary-optimum threshold statements | `theorem_local.pdf`, `theorem_global.pdf`, `theorem_band.pdf`, `theorem_ordering.pdf`, `theorem_optima.pdf` |
| `conservativeness` | Compares exact thresholds with analytical certificates across problem sizes | `conservativeness.pdf` |
| `spectral` | Studies spectral behavior as the feasibility penalty is varied | `spectral.pdf` |
| `penalty-multiplier` | Measures QAOA behavior across penalty multipliers relative to the exact global-feasibility scale | `penalty_multiplier.pdf` |
| `penalty-nx` | Studies QAOA penalty behavior across problem sizes | `penalty_nx.pdf` |
| `budget-heatmap` | Uses paired optimization checkpoints to compare QAOA iteration budgets and penalty multipliers | `budget_heatmap.pdf` |
| `optional-nx` | Compares optional-term constructions and coefficient policies across problem sizes | `optional_nx.pdf` |
| `optional-multiplier` | Sweeps optional coefficients as powers of two relative to each construction's exact ordering threshold | `optional_multiplier.pdf` |

A selected stage can be excluded from a larger selection with `--exclude-studies`:

```bash
python run_experiments.py \
  --all \
  --exclude-studies optional-multiplier \
  --out almost_all_results
```

### Optional GPU setup (custom experiments only)

The GPU setup is provided for **additional experiments**, not for the frozen paper-reproduction run.

On a Linux system with a compatible NVIDIA GPU and driver:

```bash
python setup.py --gpu
source ./venv/bin/activate
```

The GPU setup installs `requirements-gpu.txt`. To request GPU QAOA in a configurable experiment run, use:

```text
--compute-device gpu
```

> **Not for paper reproduction:** `reproduce_paper_results.py` intentionally uses `--compute-device cpu`. Installing the GPU environment does not change the frozen settings; use the CPU environment described above for the paper-production run.

## QAOA execution

The retained solver backend is PennyLane.

The main shared QAOA controls are:

| Argument | Default | Description |
|---|---:|---|
| `--qaoa-engine` | `pennylane` | QAOA implementation. The cleaned repository retains only PennyLane. |
| `--qaoa-p` | `3` | QAOA depth. |
| `--qaoa-restarts` | `3` | Number of optimization restarts for the shared QAOA studies. |
| `--qaoa-maxiter` | `100` | Maximum optimizer iterations for the shared QAOA studies. |
| `--qaoa-optimizer` | `adam` | Optimizer, either `adam` or `gradient-descent`. |
| `--qaoa-stepsize` | `0.05` | Optimizer step size. |
| `--compute-device` | `cpu` | Compute preset: `cpu`, `gpu`, or `auto`. |
| `--qaoa-device` | `auto` | Optional explicit PennyLane device override. |
| `--gpu-device-id` | `0` | GPU device index when GPU execution is used. |

For CPU execution, the automatic PennyLane device is `lightning.qubit`. For GPU execution, the automatic device is `lightning.gpu`.

With:

```text
--compute-device auto
```

the runner uses `lightning.gpu` when the GPU plugin is installed and otherwise falls back to the CPU implementation.

## CPU parallelism

Independent QAOA problem instances are executed in spawned worker processes.

The main controls are:

| Argument | Default | Description |
|---|---:|---|
| `--max-parallel-instances` | `0` | Maximum concurrent instance workers. `0` selects an automatic limit of up to four CPU workers or one GPU worker. |
| `--threads-per-worker` | `1` | Native numerical-library threads assigned to each worker. |
| `--max-tasks-per-worker` | `0` | `0` reuses workers; `1` restarts a worker after each instance for stronger memory isolation. |

The paper-production run uses:

```text
--max-parallel-instances 4
--threads-per-worker 7
```

Choose these values according to the CPU and memory available on the machine. Increasing both values aggressively can oversubscribe the system.

## Generating figures

`plot.py` contains the unified plotting implementation. It is normally called automatically by `run_experiments.py` after a successful experiment run.

To regenerate figures from CSV files that already exist in an output directory, run:

```bash
python run_experiments.py \
  --plot-only \
  --out results
```

All available manuscript figures are saved as PDF files at **300 DPI**.

The plotting code uses a headless Matplotlib workflow and does not require a graphical display. If an older version of the repository left matching PNG figure files in the result directory, those legacy PNGs are removed when the figures are regenerated.

To run experiments without generating figures at the end, use:

```text
--no-plot
```

## Output files

Each experiment output directory contains CSV data, an experiment manifest, and the figures corresponding to the selected studies.

A complete `--all` run can produce the following manuscript figures:

```text
theorem_band.pdf
theorem_local.pdf
theorem_global.pdf
theorem_ordering.pdf
theorem_optima.pdf
conservativeness.pdf
spectral.pdf
penalty_multiplier.pdf
penalty_nx.pdf
budget_heatmap.pdf
optional_nx.pdf
optional_multiplier.pdf
```

The generated data files include:

```text
theorem_thresholds.csv
theorem_sweeps.csv
ordering_sweep.csv
theorem_local.csv
theorem_global.csv
theorem_band.csv
theorem_ordering.csv
theorem_optima.csv
theorem_optima_audit.csv
theorem_optima_boundary_checks.csv
conservativeness.csv
spectral_gap.csv
qaoa_penalty.csv
scaling.csv
budget_robustness.csv
optional_nx.csv
optional_nx_paired.csv
optional_nx_audit.csv
optional_multiplier.csv
optional_multiplier_paired.csv
optional_multiplier_audit.csv
optional_seed_screen.csv
seed_ledger.csv
experiment_manifest.json
```

`experiment_manifest.json` records the selected studies, resolved QAOA device, master seed, run settings, software information, and experiment notes for the completed run.

`seed_ledger.csv` records the deterministic problem seeds used by the experiment orchestration. Optional-term studies additionally write `optional_seed_screen.csv`, which records attempted candidate seeds, exact threshold eligibility for both constructions, and screening outcomes before QAOA is run. See [Paper experiment design and instance generation](#paper-experiment-design-and-instance-generation) for the study sizes, structural sampling filters, and distinct inner/outer resampling limits.

## Reproducibility and existing outputs

The experiment runner derives deterministic problem seeds from `--master-seed` and the study/problem identifiers. The paper-production value is:

```text
--master-seed 1
```

By default, the runner refuses to overwrite a directory that already contains known generated outputs. Use a new `--out` directory when comparing runs.

If an existing generated result directory should intentionally be replaced, pass:

```text
--force
```

`--force` removes only the generated files known to the runner before starting the new experiment run.

For a clean paper-reproduction check, using a new empty output directory or removing the previous production output before launching `reproduce_paper_results.py` is recommended.

## Paper production configuration (reference only)

**Reference only.** The command below documents the exact parameters baked into `reproduce_paper_results.py`. It is **not** an alternative recommended entry point for reproducing the paper:

```bash
python run_experiments.py \
  --all \
  --out results \
  --master-seed 1 \
  \
  --instances 30 \
  --theorem-n 8 \
  \
  --conservativeness-n 6,8,10,12,14,16 \
  --conservativeness-instances-per-size 30 \
  \
  --penalty-n 6 \
  --penalty-instances 30 \
  \
  --penalty-nx-sizes 6,8,10,12,14,16 \
  --penalty-nx-instances 30 \
  --nx-restarts 3 \
  --nx-maxiter 100 \
  \
  --budget-n 6 \
  --budget-instances 30 \
  --budget-iterations 10,30,50,70,90 \
  \
  --optional-constructions random,marginal \
  --optional-nx-sizes 6,8,10,12 \
  --optional-nx-instances 30 \
  --optional-nx-restarts 3 \
  --optional-nx-maxiter 100 \
  \
  --optional-multiplier-n 8 \
  --optional-multiplier-instances 30 \
  --optional-multiplier-mode ordering-tight \
  --optional-powers=-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10 \
  --optional-multiplier-restarts 3 \
  --optional-multiplier-maxiter 100 \
  --optional-max-seed-attempts 0 \
  \
  --qaoa-engine pennylane \
  --qaoa-p 3 \
  --qaoa-restarts 3 \
  --qaoa-maxiter 100 \
  --qaoa-optimizer adam \
  --qaoa-stepsize 0.05 \
  \
  --compute-device cpu \
  --max-parallel-instances 4 \
  --threads-per-worker 7
```

**To reproduce the paper, run `python reproduce_paper_results.py` instead of manually copying this command.** The wrapper prevents accidental changes to the frozen configuration.

## Repository structure

```text
QUBO-Coefficient-Calibration/
├── reproduce_paper_results.py            # RUN THIS to reproduce the paper (frozen settings)
├── run_experiments.py                    # Additional/custom experiments (configurable)
├── plot.py                               # Unified manuscript plotting code
├── setup.py                              # Creates CPU or GPU virtual environment
├── requirements-cpu.txt                  # CPU runtime dependencies
├── requirements-gpu.txt                  # NVIDIA/CUDA 12 GPU dependencies
├── QUBO_Coefficient_Calibration/
│   ├── Problem/
│   │   └── cardinality.py                # Fixed-cardinality problem/QUBO definition
│   ├── Experiment/
│   │   ├── analysis.py                   # Shared exact enumeration, certificates, and diagnostics
│   │   ├── qaoa.py                       # Shared QAOA execution helpers
│   │   ├── theorem.py                    # Theorem and ordering studies
│   │   ├── conservativeness.py           # Certificate-conservativeness study
│   │   ├── spectral.py                   # Spectral-gap study
│   │   ├── penalty.py                    # Penalty-multiplier and penalty-Nx studies
│   │   ├── budget.py                     # Paired optimizer-budget study
│   │   └── optional.py                   # Optional-term theory, Nx, and multiplier studies
│   └── Solver/
│       ├── solver.py                     # Solver interface
│       ├── qaoa_utils.py                 # QAOA utilities
│       └── pennylane_qaoa.py             # PennyLane QAOA implementation
├── .gitignore
└── LICENSE                               # MIT License
```

## Troubleshooting

### `PennyLane not installed`

Create and activate one of the supplied environments:

```bash
python setup.py
source ./venv/bin/activate
```

or, for the Linux/NVIDIA GPU environment:

```bash
python setup.py --gpu
source ./venv/bin/activate
```

### `output already contains generated results`

The runner protects existing generated outputs from accidental replacement. Either choose a new directory:

```bash
python run_experiments.py --all --out new_results
```

or explicitly allow replacement of known generated files:

```bash
python run_experiments.py --all --out old_results --force
```

### GPU execution cannot find `lightning.gpu`

Confirm that the GPU environment was created with:

```bash
python setup.py --gpu
```

and that the machine has a compatible NVIDIA driver and CUDA 12-capable environment. The GPU preset is intended for Linux/NVIDIA systems and is rejected by `setup.py` on Windows.

### A run is using too many CPU resources

Reduce one or both of:

```text
--max-parallel-instances
--threads-per-worker
```

The total native CPU demand can approximately scale with the number of concurrent workers multiplied by the thread count assigned to each worker.

### Problem size is rejected

The current exact-enumeration/statevector workflow restricts problem sizes to:

```text
2 <= n <= 16
```

This is enforced by `run_experiments.py`.

### `--plot-only` does not produce every figure

`--plot-only` regenerates figures from the CSV files available in the selected output directory. A figure whose required data were never generated cannot be reconstructed from that directory.

### Reproduction results differ from a previous run

For comparison with the paper-production run, use the CPU environment and invoke:

```bash
python reproduce_paper_results.py
```

rather than manually modifying the experiment arguments. Also ensure that the supplied dependency environment is being used and that results are being written to a clean output directory.

## Citation

Citation information will be added when the associated paper is publicly available. Until then, please refer to this repository and the accompanying manuscript.

## License

This project is released under the [MIT License](LICENSE).
