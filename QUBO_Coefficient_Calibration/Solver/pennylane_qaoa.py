"""PennyLane statevector QAOA solver used by the cardinality experiments.

The unified runner uses a transverse mixer and normalized Ising cost
Hamiltonian.  Auxiliary tensor and mixer options remain internally supported
by this established solver, but are not separate manuscript experiments.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any, Sequence
import math
import os
import numpy as onp  # ordinary numpy for post-processing only
import pennylane as qml
import pennylane.numpy as qnp  # autograd-aware numpy

from QUBO_Coefficient_Calibration.Solver.solver import qubo_value, decode_from_qubo, qubo_value_rew, qubo_value_base
from QUBO_Coefficient_Calibration.Solver.qaoa_utils import unique_ring_edges

def _parse_pair_key(s: str) -> Tuple[int, int]:
    a, b = s.split(",")
    return (int(a), int(b))

@dataclass
class QAOASolver:
    """
    Minimal QAOA solver (gradient descent) for a vrp_problem.QUBO.
    Uses autograd-compatible qnp arrays to avoid 'no trainable parameters' warnings.
    """
    qubo: Any
    p: int = 2
    device_name: str = "default.qubit"
    shots: Optional[int] = None
    seed: Optional[int] = 0
    mixer_lambda: float = 0.5
    # Primary mixer choice. ``transverse`` implements H_M=-sum_i X_i and is
    # the clean coefficient-sensitivity baseline; ``xy`` preserves the manuscript
    # entangling ring mixer for supplementary robustness experiments.
    mixer_kind: str = "xy"  # {"transverse", "xy"}
    # Classical optimizer used for the variational angles.  ``gradient-descent``
    # preserves the historical update rule; ``adam`` uses the usual bias-corrected
    # first/second-moment updates while retaining the same robust gradient
    # evaluation/fallback path.
    optimizer_kind: str = "gradient-descent"  # {"gradient-descent", "adam"}
    # The manuscript defines the ideal unitary exp(-i theta H_mixer).
    # PennyLane's qaoa.mixer_layer silently uses a one-step first-order
    # Trotterization for a general (noncommuting) Hamiltonian, so do not use
    # that helper implicitly.  ``trotter`` is the memory-friendly simulator
    # path; ``exact`` requests PennyLane's analytic Evolution operator and is
    # intended only for small systems/backends that support it efficiently.
    mixer_evolution: str = "trotter"  # {"trotter", "exact"}
    mixer_trotter_steps: int = 4

    # Quantum-circuit simulation backend.  ``statevector`` preserves the previous
    # default.qubit/lightning.gpu behavior.  ``tensor`` uses an MPS/TN backend:
    # default.tensor (CPU/quimb) or lightning.tensor (NVIDIA/cuTensorNet).
    simulator: str = "statevector"  # {"statevector", "tensor"}
    tensor_method: str = "mps"       # {"mps", "tn"}; mps can be approximate
    tensor_max_bond_dim: Optional[int] = 64
    tensor_cutoff: float = 1e-10
    tensor_cutoff_mode: str = "abs"  # lightning.tensor only; {"abs", "rel"}

    # Optional positive rescaling of the Ising cost Hamiltonian.  When enabled,
    # all nonconstant Ising coefficients are divided by kappa=max(|h|,|J|).
    # This leaves every basis-state ordering unchanged while putting different
    # penalty regimes on a comparable QAOA angle/frequency scale.
    normalize_cost: bool = False

    # Derived / internal
    n_qubits: int = field(init=False)
    c0: float = field(init=False)  # constant QUBO/Ising offset; never trainable
    cost_hamiltonian: Any = field(init=False)
    dev: Any = field(init=False)
    readout_dev: Any = field(init=False, default=None)
    params: qnp.ndarray = field(init=False)  # shape (2p,), trainable
    qnode_diff_method: Optional[str] = field(init=False, default=None)
    candidate_tracking: str = field(init=False, default="per_iteration")
    cost_scale: float = field(init=False, default=1.0)
    optimization_offset: float = field(init=False, default=0.0)

    def __post_init__(self):
        if int(self.p) <= 0:
            raise ValueError("QAOA depth p must be positive.")
        self.p = int(self.p)
        self.mixer_lambda = float(self.mixer_lambda)
        self.mixer_kind = str(self.mixer_kind).lower().strip()
        if self.mixer_kind not in {"transverse", "xy"}:
            raise ValueError("mixer_kind must be 'transverse' or 'xy'.")
        self.optimizer_kind = str(self.optimizer_kind).lower().strip().replace("_", "-")
        if self.optimizer_kind == "gd":
            self.optimizer_kind = "gradient-descent"
        if self.optimizer_kind not in {"gradient-descent", "adam"}:
            raise ValueError("optimizer_kind must be 'gradient-descent' or 'adam'.")
        if not math.isfinite(self.mixer_lambda) or self.mixer_lambda <= 0.0:
            raise ValueError("mixer_lambda must be finite and > 0.")
        self.mixer_evolution = str(self.mixer_evolution).lower().strip()
        if self.mixer_evolution not in {"trotter", "exact"}:
            raise ValueError("mixer_evolution must be 'trotter' or 'exact'.")
        self.mixer_trotter_steps = int(self.mixer_trotter_steps)
        if self.mixer_trotter_steps <= 0:
            raise ValueError("mixer_trotter_steps must be >= 1.")

        self.simulator = str(self.simulator).lower().strip()
        if self.simulator not in {"statevector", "tensor"}:
            raise ValueError("simulator must be 'statevector' or 'tensor'.")
        self.tensor_method = str(self.tensor_method).lower().strip()
        if self.tensor_method not in {"mps", "tn"}:
            raise ValueError("tensor_method must be 'mps' or 'tn'.")
        if self.tensor_max_bond_dim is not None:
            self.tensor_max_bond_dim = int(self.tensor_max_bond_dim)
            if self.tensor_max_bond_dim <= 0:
                self.tensor_max_bond_dim = None
        self.tensor_cutoff = float(self.tensor_cutoff)
        if not math.isfinite(self.tensor_cutoff) or self.tensor_cutoff < 0.0:
            raise ValueError("tensor_cutoff must be finite and >= 0.")
        self.tensor_cutoff_mode = str(self.tensor_cutoff_mode).lower().strip()
        self.normalize_cost = bool(self.normalize_cost)
        if self.tensor_cutoff_mode not in {"abs", "rel"}:
            raise ValueError("tensor_cutoff_mode must be 'abs' or 'rel'.")

        if self.simulator == "tensor":
            if self.device_name not in {"default.tensor", "lightning.tensor"}:
                raise ValueError(
                    "tensor simulation requires device_name='default.tensor' (CPU) "
                    "or 'lightning.tensor' (NVIDIA GPU)."
                )
            if self.mixer_kind == "xy" and self.mixer_evolution != "trotter":
                raise ValueError(
                    "tensor simulation with the XY mixer currently requires mixer_evolution='trotter' "
                    "so the noncommuting ring mixer is decomposed into tensor-backend-supported gates."
                )
            if self.shots is not None and self.device_name == "default.tensor":
                raise ValueError("default.tensor supports analytic execution only (shots=None).")
            # Tensor devices support parameter-shift reliably, whereas backprop/adjoint
            # is not generally available.  Also avoid materializing a dense probability
            # vector at every optimizer step; perform one dense readout after optimization.
            self.qnode_diff_method = "parameter-shift"
            self.candidate_tracking = "final_only"
        else:
            # Lightning statevector backends expose an efficient adjoint gradient
            # method.  In particular this prevents GPU QAOA from falling back to
            # many parameter-shift circuit evaluations per optimizer step.
            if self.device_name in {"lightning.gpu", "lightning.qubit"}:
                self.qnode_diff_method = "adjoint"
            else:
                self.qnode_diff_method = None
            self.candidate_tracking = "per_iteration"

        # The manuscript maps every QUBO binary (all x and y bits) to one
        # qubit, even if a collected coefficient happens to cancel to zero.
        self.n_qubits = len(self.qubo.x_index) + len(self.qubo.y_index)
        if self.n_qubits <= 0:
            raise ValueError("QAOA requires at least one QUBO binary variable.")

        # Convert to Ising and build the optimization Hamiltonian.  The QUBO
        # constant is kept separate because it never affects the circuit.  For
        # the coefficient-sensitivity experiments we optionally divide every Z
        # and ZZ coefficient by kappa=max(|h|,|J|); positive rescaling preserves
        # the complete ground-state ordering while avoiding a trivial frequency-
        # scale change as penalty coefficients grow.
        ising = self.qubo.to_ising()
        self.c0 = float(ising["offset"])

        lin = {int(k): float(v) for k, v in ising["linear"].items()}
        quad = {_parse_pair_key(k): float(v) for k, v in ising["quadratic"].items()}
        magnitudes = [abs(v) for v in lin.values()] + [abs(v) for v in quad.values()]
        kappa = max(magnitudes, default=0.0)
        if not math.isfinite(kappa):
            raise ValueError("Ising coefficient scale is not finite.")
        self.cost_scale = float(kappa if self.normalize_cost and kappa > 0.0 else 1.0)
        self.optimization_offset = float(self.c0 / self.cost_scale)

        coeffs, ops = [], []
        for i, h in lin.items():
            if abs(h) > 0.0:
                coeffs.append(h / self.cost_scale); ops.append(qml.PauliZ(i))
        for (i, j), J in quad.items():
            if abs(J) > 0.0:
                coeffs.append(J / self.cost_scale); ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
        if not coeffs:
            coeffs = [0.0]; ops = [qml.PauliZ(0)]
        self.cost_hamiltonian = qml.Hamiltonian(coeffs, ops)

        # Device.  Tensor simulators need backend-specific construction arguments.
        # The tensor readout QNode receives its own device because lightning.tensor
        # keeps graph data attached to a device instance.
        self.dev = self._make_device()
        self.readout_dev = self._make_device() if self.simulator == "tensor" else self.dev

        # Initialize params [gammas (p), betas (p)] as qnp arrays with requires_grad=True
        rng = onp.random.default_rng(self.seed)
        init = onp.concatenate([
            rng.uniform(0.0, onp.pi, size=self.p),
            rng.uniform(0.0, onp.pi, size=self.p),
        ])
        self.params = qnp.array(init, requires_grad=True)

        # Build QNodes
        self._build_qnodes()

    def _make_device(self):
        """Construct the requested PennyLane simulator without hiding backend errors."""
        kwargs = {"wires": self.n_qubits}
        if self.simulator == "tensor":
            # default.tensor is analytic-only and does not expose shots as a
            # device option; lightning.tensor supports shots, although this
            # runner uses analytic expectation values for optimization.
            if self.device_name == "lightning.tensor":
                kwargs["shots"] = self.shots
            kwargs["method"] = self.tensor_method
            if self.tensor_method == "mps":
                kwargs["max_bond_dim"] = self.tensor_max_bond_dim
                kwargs["cutoff"] = self.tensor_cutoff
                if self.device_name == "lightning.tensor":
                    kwargs["cutoff_mode"] = self.tensor_cutoff_mode
            return qml.device(self.device_name, **kwargs)

        # Seed statevector simulators/samplers where supported.
        kwargs["shots"] = self.shots
        if self.device_name == "lightning.qubit":
            # Lightning-Qubit wheels provide OpenMP-threaded kernels on Linux/macOS.
            # batch_obs also allows adjoint differentiation of Hamiltonian terms to
            # use the OpenMP team when multiple observable contributions are present.
            try:
                omp_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
            except ValueError:
                omp_threads = 1
            kwargs["batch_obs"] = bool(omp_threads > 1)
        try:
            return qml.device(self.device_name, seed=self.seed, **kwargs)
        except TypeError:
            # Some plugin/version combinations do not expose ``seed`` at device
            # construction. Keep the backend options (including batch_obs).
            return qml.device(self.device_name, **kwargs)

    # --------------------- QNodes ---------------------
    # def _layered_ansatz(self, theta):
    #     gammas = theta[: self.p]
    #     betas  = theta[self.p :]

    #     # Start in |+>^n
    #     for w in range(self.n_qubits):
    #         qml.Hadamard(wires=w)

    #     # p layers
    #     for l in range(self.p):
    #         qml.qaoa.cost_layer(gammas[l], self.cost_hamiltonian)
    #         # Mixer: RX(2*beta) on every qubit (exp(-i * beta * X))
    #         for w in range(self.n_qubits):
    #             qml.RX(2.0 * betas[l], wires=w)

    # def _entangling_x_mixer_h(self, edges=None, lam_xx=0.5):
    #     """
    #     Build H_mix = sum_i X_i  + lam_xx * sum_(i,j in edges) X_i X_j
    #     - edges: list of (i,j) pairs; defaults to a ring if None
    #     - lam_xx: weight on the XX entanglers (tune if you like)
    #     """
    #     if edges is None:
    #         # hardware-friendly ring
    #         edges = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]

    #     coeffs = [1.0] * self.n_qubits
    #     ops    = [qml.PauliX(i) for i in range(self.n_qubits)]

    #     for i, j in edges:
    #         coeffs.append(lam_xx)
    #         ops.append(qml.PauliX(i) @ qml.PauliX(j))

    #     return qml.Hamiltonian(coeffs, ops)

    # def _layered_ansatz(self, theta, *, lam_xx=0.5, edges=None, warm_start_angles=None):
    #     """
    #     Better QAOA ansatz:
    #     - Cost layer:     exp(-i gamma_l H_cost)
    #     - Entangling mixer: exp(-i beta_l (sum X_i + lam_xx * sum X_i X_j))
    #     - Optional warm-start: initialize with RY(2*phi_i) instead of |+>^n

    #     Args:
    #     theta: shape (2p,) -> gammas then betas
    #     lam_xx: strength of the XX part of the mixer
    #     edges: connectivity for XX terms (defaults to ring)
    #     warm_start_angles: optional list/array of per-qubit angles phi_i for RY(2*phi_i)
    #     """
    #     p = self.p
    #     gammas = theta[:p]
    #     betas  = theta[p:2*p]

    #     # --- Initialization ---
    #     if warm_start_angles is not None:
    #         # warm start (e.g., from a classical solution in [0, pi/2])
    #         for w in range(self.n_qubits):
    #             qml.RY(2.0 * float(warm_start_angles[w]), wires=w)
    #     else:
    #         # |+>^n
    #         for w in range(self.n_qubits):
    #             qml.Hadamard(wires=w)

    #     # --- Mixer Hamiltonian with entanglement ---
    #     H_mix = self._entangling_x_mixer_h(edges=edges, lam_xx=lam_xx)

    #     # --- p alternating layers ---
    #     for l in range(p):
    #         qml.qaoa.cost_layer(gammas[l], self.cost_hamiltonian)
    #         qml.qaoa.mixer_layer(betas[l], H_mix)

    def _build_paper_mixer(self):
        """
        Improved mixer:
            H_mixer = -∑_i X_i  - λ ∑_i (X_i X_{i+1} + Y_i Y_{i+1})

        - First term is the original transverse field.
        - Second term is an XY-type ring mixer that entangles neighbours.
        """
        coeffs = []
        ops = []

        # ----- 1) Original transverse-field part: -∑_i X_i -----
        for i in range(self.n_qubits):
            coeffs.append(-1.0)
            ops.append(qml.PauliX(i))

        # ----- 2) Add XY ring couplings: -λ (X_i X_{i+1} + Y_i Y_{i+1}) -----
        # Manuscript experiment setting: lambda = 0.5 by default.
        lam = float(self.mixer_lambda)

        # R_n contains each *undirected* physical ring edge exactly once.
        # This matters for n=2 (one edge, not two opposite orientations) and
        # n=1 (no self-edge), exactly as stated in the manuscript.
        for i, j in unique_ring_edges(self.n_qubits):
            coeffs.append(-lam)
            ops.append(qml.PauliX(i) @ qml.PauliX(j))

            coeffs.append(-lam)
            ops.append(qml.PauliY(i) @ qml.PauliY(j))

        self._H_mixer_paper = qml.Hamiltonian(coeffs, ops)


    def _apply_transverse_mixer(self, beta):
        """Apply exp(-i beta H_M) for H_M=-sum_i X_i exactly as local RX gates."""
        # RX(theta)=exp(-i theta X/2), hence theta=-2*beta for H_M=-X.
        for w in range(self.n_qubits):
            qml.RX(-2.0 * beta, wires=w)

    def _apply_paper_mixer(self, beta):
        """Apply the manuscript mixer evolution.

        ``exact`` queues PennyLane's symbolic Evolution operator
        exp(-i beta H_mixer).  This is the literal manuscript unitary, but a
        generic backend may realize it through an expensive matrix exponential.

        ``trotter`` uses an explicit first-order Suzuki--Trotter approximation
        with ``mixer_trotter_steps`` steps.  This is the practical low-memory
        statevector path.  It is intentionally explicit because
        qml.qaoa.mixer_layer(...) otherwise hard-codes one Trotter step while
        presenting itself as the ideal exponential.
        """
        if not hasattr(self, "_H_mixer_paper"):
            self._build_paper_mixer()
        if self.mixer_evolution == "exact":
            qml.evolve(self._H_mixer_paper, coeff=beta)
        else:
            qml.ApproxTimeEvolution(
                self._H_mixer_paper, beta, int(self.mixer_trotter_steps)
            )

    def _layered_ansatz(self, theta):
        p = self.p
        gammas = theta[:p]
        betas  = theta[p:]

        # Manuscript initialization: |+>^n.
        for w in range(self.n_qubits):
            qml.Hadamard(wires=w)

        for l in range(p):
            # cost evolution block e^{-i gamma_l H_cost}
            qml.qaoa.cost_layer(gammas[l], self.cost_hamiltonian)
            # The primary coefficient-sensitivity experiment uses the ordinary
            # transverse-field mixer; the XY ring mixer remains selectable for
            # supplementary robustness checks.
            if self.mixer_kind == "transverse":
                self._apply_transverse_mixer(betas[l])
            else:
                self._apply_paper_mixer(betas[l])

    def _build_qnodes(self):
        qnode_kwargs = {"interface": "autograd"}
        if self.qnode_diff_method is not None:
            qnode_kwargs["diff_method"] = self.qnode_diff_method

        @qml.qnode(self.dev, **qnode_kwargs)
        def expval_cost(theta):
            self._layered_ansatz(theta)
            return qml.expval(self.cost_hamiltonian)

        self._expval_cost = expval_cost

        if self.simulator == "tensor":
            # default.tensor does not provide a finite-shot/probability workflow
            # suitable for our CPU path.  Materialize the dense state only once,
            # after angle optimization, and convert it to probabilities.  This keeps
            # the iterative optimization tensor-network based while preserving the
            # same top-k readout semantics as the statevector implementation.
            @qml.qnode(self.readout_dev, interface="autograd")
            def state_circuit(theta):
                self._layered_ansatz(theta)
                return qml.state()

            self._state_circuit = state_circuit

            def probs_circuit(theta):
                state = onp.asarray(qml.math.toarray(self._state_circuit(theta)))
                probs = onp.abs(state) ** 2
                total = float(onp.sum(probs))
                if not math.isfinite(total) or total <= 0.0:
                    raise FloatingPointError("Tensor-network final state has invalid norm.")
                return probs / total

            self._probs_circuit = probs_circuit
        else:
            @qml.qnode(self.dev, interface="autograd")
            def probs_circuit(theta):
                self._layered_ansatz(theta)
                return qml.probs(wires=range(self.n_qubits))

            self._probs_circuit = probs_circuit

    # ------------------ Public API ------------------
    def cost(self, theta):
        # IMPORTANT: do NOT cast to float here; keep it as qnp scalar so autograd sees trainable params
        return self._expval_cost(theta) + self.optimization_offset

    # def optimize(self, maxiter: int = 200, stepsize: float = 0.2, tol: float = 1e-6, verbose: bool = False):
    #     opt = qml.GradientDescentOptimizer(stepsize=stepsize)
    #     theta = qnp.array(self.params, requires_grad=True)  # keep trainable
    #     history = []
    #     prev = None
    #     for it in range(maxiter):
    #         theta, val = opt.step_and_cost(self.cost, theta)
    #         # 'val' is a Python float from PennyLane; okay to store
    #         history.append(val)
    #         if verbose and (it % max(1, maxiter // 10) == 0):
    #             print(f"[{it:04d}] E = {val:.8f}")
    #         if prev is not None and abs(prev - val) < tol:
    #             break
    #         prev = val
    #     self.params = theta
    #     return {"params": theta, "energy": float(self.cost(theta)), "history": history}

    @staticmethod
    def _to_float(value) -> float:
        """Convert a PennyLane/Autograd scalar to an ordinary Python float."""
        try:
            return float(qml.math.toarray(value))
        except Exception:
            return float(value)

    @staticmethod
    def _as_trainable(theta):
        """Detach an angle vector from any old Autograd trace and mark it trainable.

        PennyLane optimizers may return a tensor whose ``requires_grad`` metadata
        is lost after an update in some version/interface combinations.  QAOA only
        needs gradients for the *current* angle vector, so detaching between
        iterations is both safe and avoids that failure mode.
        """
        return qnp.array(onp.asarray(theta, dtype=float), requires_grad=True)

    def _finite_difference_gradient(self, theta, rel_step: float = 1e-6):
        """Deterministic central finite-difference gradient fallback.

        This is intentionally a fallback rather than the primary optimizer.  It
        keeps the QAOA experiment usable if PennyLane/Autograd raises
        ``NonDifferentiableError`` despite receiving an explicitly trainable
        angle vector.  With p=1 in the manuscript experiment this costs only four
        extra expectation-value evaluations per optimization step.
        """
        x = onp.asarray(theta, dtype=float).copy()
        grad = onp.zeros_like(x, dtype=float)
        for j in range(x.size):
            h = float(rel_step * max(1.0, abs(x[j])))
            xp = x.copy(); xp[j] += h
            xm = x.copy(); xm[j] -= h
            fp = self._to_float(self.cost(qnp.array(xp, requires_grad=False)))
            fm = self._to_float(self.cost(qnp.array(xm, requires_grad=False)))
            grad[j] = (fp - fm) / (2.0 * h)
        return grad

    def optimize(
        self,
        maxiter: int = 200,
        stepsize: float = 0.2,
        tol: float = 1e-6,
        top_k: int = 50,
        verbose: bool = False,
        checkpoint_iterations: Optional[Sequence[int]] = None,
    ):
        """Optimize QAOA angles with a robust PennyLane/Autograd loop.

        The old implementation delegated updates to
        ``qml.GradientDescentOptimizer.step_and_cost``.  On PennyLane 0.43 this
        can surface ``NonDifferentiableError`` on a later iteration when the
        returned parameter tensor no longer carries trainability metadata.

        Here each iteration explicitly detaches/re-wraps the angle vector with
        ``requires_grad=True`` before differentiating.  If PennyLane still raises
        ``NonDifferentiableError`` for that iteration, a deterministic central
        finite-difference gradient is used instead.  Other exceptions are not
        hidden and still propagate normally.
        """
        theta = self._as_trainable(self.params)
        grad_fn = qml.grad(self.cost)

        history = []
        per_iter_bests = []
        checkpoint_set = {int(v) for v in (checkpoint_iterations or []) if int(v) > 0}
        checkpoint_snapshots = {}
        global_best_bits = None
        global_best_val = math.inf
        gradient_fallback_count = 0
        gradient_methods = []

        # Adam state.  These constants are the standard Adam defaults; ``stepsize``
        # remains the user-controlled learning rate for either optimizer.
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_eps = 1e-8
        adam_m = onp.zeros_like(onp.asarray(theta, dtype=float))
        adam_v = onp.zeros_like(onp.asarray(theta, dtype=float))

        prev = None
        for it in range(maxiter):
            # Important: create a fresh leaf tensor every iteration.  We do not
            # need gradients through previous optimization steps.
            theta = self._as_trainable(theta)

            try:
                grad = grad_fn(theta)
                grad_np = onp.asarray(qml.math.toarray(grad), dtype=float)
                if self.qnode_diff_method == "parameter-shift":
                    gradient_method = "parameter_shift"
                elif self.qnode_diff_method == "adjoint":
                    gradient_method = "adjoint"
                else:
                    gradient_method = "autograd"
            except Exception as exc:
                # Do not silently swallow unrelated QNode/device errors.
                if exc.__class__.__name__ != "NonDifferentiableError":
                    raise
                grad_np = self._finite_difference_gradient(theta)
                gradient_fallback_count += 1
                gradient_method = "finite_difference_fallback"

            if grad_np.shape != onp.asarray(theta).shape:
                grad_np = onp.reshape(grad_np, onp.asarray(theta).shape)
            if not onp.all(onp.isfinite(grad_np)):
                raise FloatingPointError(f"QAOA gradient became non-finite at iteration {it}: {grad_np!r}")

            # Match GradientDescentOptimizer semantics: history/stopping uses the
            # objective value *before* the update, while candidate bitstrings are
            # evaluated at the newly updated angles.
            val = self._to_float(self.cost(theta))
            history.append(val)
            gradient_methods.append(gradient_method)

            theta_arr = onp.asarray(theta, dtype=float)
            if self.optimizer_kind == "gradient-descent":
                theta_np = theta_arr - float(stepsize) * grad_np
            else:
                t = it + 1
                adam_m = adam_beta1 * adam_m + (1.0 - adam_beta1) * grad_np
                adam_v = adam_beta2 * adam_v + (1.0 - adam_beta2) * (grad_np ** 2)
                m_hat = adam_m / (1.0 - adam_beta1 ** t)
                v_hat = adam_v / (1.0 - adam_beta2 ** t)
                theta_np = theta_arr - float(stepsize) * m_hat / (onp.sqrt(v_hat) + adam_eps)
            theta = self._as_trainable(theta_np)

            # A work budget of B means the variational state *after B optimizer
            # updates*.  Store the angle vector and objective at that point so a
            # single longest run can provide paired checkpoints for the budget
            # heatmap without restarting Adam or changing initialization seeds.
            step_number = it + 1
            if step_number in checkpoint_set:
                checkpoint_snapshots[step_number] = {
                    "params": onp.asarray(theta, dtype=float).copy(),
                    "expval_energy_optimization_scale": self._to_float(self.cost(theta)),
                    "iterations_completed": int(step_number),
                }

            best_bits_it = None
            best_val_it = math.inf
            best_prob_it = None
            if self.candidate_tracking == "per_iteration":
                probs = self._probs_circuit(theta)
                cands = self._top_k_bitstrings_from_probs(probs, k=top_k)
                for bits, p in cands:
                    v = qubo_value(self.qubo, bits)
                    if v < best_val_it:
                        best_val_it = v
                        best_bits_it = bits
                        best_prob_it = p
                if best_val_it < global_best_val:
                    global_best_val = best_val_it
                    global_best_bits = best_bits_it

            per_iter_bests.append({
                "iter": it,
                "expval_energy": float(val),
                "best_bits": best_bits_it,
                "best_energy": (float(best_val_it) if best_bits_it is not None else None),
                "best_prob": float(best_prob_it) if best_prob_it is not None else None,
                "gradient_method": gradient_method,
            })

            if verbose and (it % max(1, maxiter // 10) == 0):
                best_txt = f"{best_val_it:.8f}" if best_bits_it is not None else "deferred"
                print(
                    f"[{it:04d}] E = {val:.8f} | "
                    f"best_bits_energy = {best_txt} | grad={gradient_method}"
                )

            if prev is not None and abs(prev - val) < tol:
                break
            prev = val

        self.params = self._as_trainable(theta)
        completed_steps = len(history)
        # If the optimizer satisfies its stopping criterion before a requested
        # checkpoint, later budgets represent the same converged state.  This
        # preserves the meaning of "run once, inspect at larger budgets" rather
        # than silently launching a new optimization trajectory.
        if checkpoint_set:
            final_params = onp.asarray(self.params, dtype=float).copy()
            final_energy = self._to_float(self.cost(self.params))
            for step_number in sorted(checkpoint_set):
                if step_number not in checkpoint_snapshots:
                    checkpoint_snapshots[step_number] = {
                        "params": final_params.copy(),
                        "expval_energy_optimization_scale": float(final_energy),
                        "iterations_completed": int(completed_steps),
                    }

        return {
            "params": self.params,
            "energy": float(self.cost(self.params)),
            "history": history,
            "per_iter_bests": per_iter_bests,
            "global_best_bits": global_best_bits,
            "global_best_energy": float(global_best_val),
            "gradient_methods": gradient_methods,
            "gradient_fallback_count": int(gradient_fallback_count),
            "checkpoint_snapshots": checkpoint_snapshots,
            "optimizer_iterations_completed": int(completed_steps),
            "optimizer_kind": self.optimizer_kind,
            "optimizer_stepsize": float(stepsize),
            "gradient_method": (
                "finite_difference_fallback"
                if gradient_fallback_count
                else (
                    "parameter_shift" if self.qnode_diff_method == "parameter-shift"
                    else ("adjoint" if self.qnode_diff_method == "adjoint" else "autograd")
                )
            ),
        }

    def _top_k_bitstrings_from_probs(self, probs, k: int = 50):
        # Work in ordinary numpy for indexing/sorting
        probs = onp.array(probs)
        k = min(k, len(probs))
        idxs = onp.argpartition(-probs, k - 1)[:k]
        idxs = idxs[onp.argsort(-probs[idxs])]
        out = []
        for idx in idxs:
            p = probs[idx]
            bits = onp.array(list(map(int, onp.binary_repr(idx, width=self.n_qubits))))
            out.append((bits, float(p)))
        return out

    # def solve(self, maxiter: int = 200, stepsize: float = 0.2, tol: float = 1e-6, top_k: int = 50, verbose: bool = False):
    #     opt = self.optimize(maxiter=maxiter, stepsize=stepsize, tol=tol, verbose=verbose)
    #     theta = opt["params"]
    #     expval_energy = float(self.cost(theta))

    #     # Full probability distribution (analytic mode if shots=None)
    #     probs = self._probs_circuit(theta)
    #     cands = self._top_k_bitstrings_from_probs(probs, k=top_k)

    #     best_bits = None
    #     best_val = math.inf
    #     for bits, _p in cands:
    #         val = qubo_value(bits)
    #         if val < best_val:
    #             best_val = val
    #             best_bits = bits

    #     decoded = decode_from_qubo(best_bits)

    #     return {
    #         "optimal_params": onp.array(theta),
    #         "expval_energy": expval_energy,
    #         "best_bits": best_bits,
    #         "best_energy": float(best_val),
    #         "decoded": decoded,
    #         "history": opt["history"],
    #     }
        
    def solve(
        self,
        maxiter: int = 200,
        stepsize: float = 0.2,
        tol: float = 1e-6,
        top_k: int = 50,
        verbose: bool = False,
        return_probabilities: bool = False,
        checkpoint_iterations: Optional[Sequence[int]] = None,
    ):
        # Pass top_k through so optimize evaluates per-step bests.  Checkpoint
        # snapshots are taken inside the uninterrupted optimization loop so Adam
        # moments and all restart initializations are shared across budgets.
        opt = self.optimize(
            maxiter=maxiter, stepsize=stepsize, tol=tol, top_k=top_k, verbose=verbose,
            checkpoint_iterations=checkpoint_iterations,
        )
        theta = opt["params"]
        expval_optimization = float(self.cost(theta))
        # Convert the normalized expectation back to the original QUBO energy
        # scale for backward-compatible reporting.
        expval_energy = float(expval_optimization * self.cost_scale)
        final_probs = None
        if return_probabilities:
            final_probs = onp.asarray(self._probs_circuit(theta), dtype=float)
            total = float(onp.sum(final_probs))
            if not math.isfinite(total) or total <= 0.0:
                raise FloatingPointError("Final QAOA probability vector has invalid norm.")
            final_probs = final_probs / total

        checkpoint_results = {}
        for step_number, snap in sorted(opt.get("checkpoint_snapshots", {}).items()):
            checkpoint_theta = qnp.array(snap["params"], requires_grad=False)
            checkpoint_probs = onp.asarray(self._probs_circuit(checkpoint_theta), dtype=float)
            checkpoint_total = float(onp.sum(checkpoint_probs))
            if not math.isfinite(checkpoint_total) or checkpoint_total <= 0.0:
                raise FloatingPointError(f"Checkpoint {step_number} probability vector has invalid norm.")
            checkpoint_probs = checkpoint_probs / checkpoint_total
            checkpoint_results[int(step_number)] = {
                "optimal_params": onp.asarray(snap["params"], dtype=float).copy(),
                "expval_energy_optimization_scale": float(snap["expval_energy_optimization_scale"]),
                "expval_energy": float(snap["expval_energy_optimization_scale"] * self.cost_scale),
                "final_probabilities": checkpoint_probs,
                "iterations_completed": int(snap["iterations_completed"]),
                "cost_scale_kappa": float(self.cost_scale),
            }

        # Prefer the global best found during the run
        if opt["global_best_bits"] is not None:
            best_bits = opt["global_best_bits"]
            best_val = opt["global_best_energy"]
        else:
            # Fallback: compute from final parameters (original behavior)
            probs = final_probs if final_probs is not None else self._probs_circuit(theta)
            cands = self._top_k_bitstrings_from_probs(probs, k=top_k)
            best_bits, best_val = None, float("inf")
            for bits, _p in cands:
                v = qubo_value(self.qubo, bits)
                if v < best_val:
                    best_val = v
                    best_bits = bits

        decoded = decode_from_qubo(self.qubo, best_bits)
        best_val_rew = qubo_value_rew(self.qubo, best_bits)
        best_val_base = qubo_value_base(self.qubo, best_bits)

        return {
            "optimal_params": onp.array(theta),
            "expval_energy": expval_energy,
            "expval_energy_optimization_scale": expval_optimization,
            "cost_normalized": bool(self.normalize_cost),
            "cost_scale_kappa": float(self.cost_scale),
            "final_probabilities": final_probs,
            "best_bits": best_bits,
            "best_energy": float(best_val),
            "best_energy_rew": float(best_val_rew),
            "best_energy_base": float(best_val_base),
            "seed": int(self.seed) if self.seed is not None else None,
            "device_name": str(self.device_name),
            "simulator": self.simulator,
            "tensor_method": self.tensor_method if self.simulator == "tensor" else None,
            "tensor_max_bond_dim": (
                self.tensor_max_bond_dim
                if self.simulator == "tensor" and self.tensor_method == "mps"
                else None
            ),
            "tensor_cutoff": (
                float(self.tensor_cutoff)
                if self.simulator == "tensor" and self.tensor_method == "mps"
                else None
            ),
            "tensor_cutoff_mode": (
                self.tensor_cutoff_mode
                if self.simulator == "tensor" and self.tensor_method == "mps"
                else None
            ),
            "tensor_approximate": bool(
                self.simulator == "tensor"
                and self.tensor_method == "mps"
                and (self.tensor_max_bond_dim is not None or self.tensor_cutoff > 0.0)
            ),
            "candidate_tracking": self.candidate_tracking,
            "final_readout": ("dense_state_once" if self.simulator == "tensor" else "dense_probs"),
            "decoded": decoded,
            "history": opt["history"],
            "per_iter_bests": opt["per_iter_bests"],             # full per-iter record
            "global_best_bits": opt["global_best_bits"],         # convenience
            "global_best_energy": opt["global_best_energy"],     # convenience
            "gradient_method": opt["gradient_method"],
            "gradient_methods": opt["gradient_methods"],
            "gradient_fallback_count": opt["gradient_fallback_count"],
            "checkpoint_results": checkpoint_results,
            "optimizer_iterations_completed": opt.get("optimizer_iterations_completed", len(opt["history"])),
            "mixer_kind": self.mixer_kind,
            "optimizer_kind": self.optimizer_kind,
            "optimizer_stepsize": float(stepsize),
            "mixer_lambda": float(self.mixer_lambda) if self.mixer_kind == "xy" else None,
            "mixer_evolution": self.mixer_evolution if self.mixer_kind == "xy" else "exact_local_rx",
            "mixer_trotter_steps": (
                int(self.mixer_trotter_steps) if self.mixer_kind == "xy" and self.mixer_evolution == "trotter" else None
            ),
        }
        
    def exact_ising_ground_state(self, max_n: int = 26, return_qubo: bool = True):
        """
        Compute the exact ground state of the Ising Hamiltonian built from the QUBO.
        Because H has only Z and ZZ terms, it's diagonal in the Z basis; enumerate 2^n states.

        Parameters
        ----------
        max_n : int
            Safety cap on number of qubits (2^n states). Keep small in practice.
        return_qubo : bool
            Also compute and return the QUBO value and decoded variables for the ground state.

        Returns
        -------
        dict with keys:
          - 'ground_bits' : np.ndarray of {0,1}^n
          - 'ground_spins': np.ndarray of {+1,-1}^n
          - 'ground_energy_ising': float
          - 'ground_energy_qubo': float (if return_qubo=True)
          - 'decoded' : dict with x/y maps (if return_qubo=True)
        """
        n = self.n_qubits
        if n > max_n:
            raise ValueError(f"n={n} exceeds max_n={max_n}. Reduce problem size or raise max_n.")

        # Use Ising dicts directly
        ising = self.qubo.to_ising()
        c0 = float(ising["offset"])
        lin = {int(k): float(v) for k, v in ising["linear"].items()}
        quad = {tuple(map(int, k.split(","))): float(v) for k, v in ising["quadratic"].items()}

        h = onp.zeros(n, dtype=float)
        for i, v in lin.items():
            h[i] = v

        pairs = list(quad.keys())
        Jvals = [quad[p] for p in pairs]

        best_E = float("inf")
        best_bits = None
        best_spins = None

        total = 1 << n
        for state in range(total):
            bits = onp.array(list(map(int, onp.binary_repr(state, width=n))), dtype=int)
            z = 1 - 2 * bits  # {0,1} -> {+1,-1}
            E = c0 + float(h @ z)
            for idx, (i, j) in enumerate(pairs):
                E += Jvals[idx] * z[i] * z[j]
            if E < best_E:
                best_E = E
                best_bits = bits
                best_spins = z

        result = {
            "ground_bits": best_bits,
            "ground_spins": best_spins,
            "ground_energy_ising": float(best_E),
        }
        if return_qubo:
            qval = qubo_value(self.qubo, best_bits)
            qval_rew = qubo_value_rew(self.qubo, best_bits)
            decoded = decode_from_qubo(self.qubo, best_bits)
            result.update({"ground_energy_qubo": float(qval), "ground_energy_qubo_rew": float(qval_rew), "decoded": decoded})
        return result