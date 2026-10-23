from __future__ import annotations
from typing import Dict, Any
import numpy as np


def qubo_value(qubo: Any, bits: np.ndarray) -> float:
    """Compute offset + sum h_i x_i + sum_{i<j} J_ij x_i x_j for bitstring bits."""
    x = bits
    val = float(qubo.offset)
    for i, h in qubo.linear.items():
        val += float(h) * float(x[i])
    for (i, j), J in qubo.quadratic.items():
        val += float(J) * float(x[i]) * float(x[j])
    return val

def qubo_value_rew(qubo: Any, bits: np.ndarray) -> float:
    """Compute offset + sum h_i x_i + sum_{i<j} J_ij x_i x_j for bitstring bits."""
    x = bits
    val = float(qubo.offset_rew)
    for i, h in qubo.linear_rew.items():
        val += float(h) * float(x[i])
    for (i, j), J in qubo.quadratic_rew.items():
        val += float(J) * float(x[i]) * float(x[j])
    return val

def qubo_value_base(qubo: Any, bits: np.ndarray) -> float:
    """Compute offset + sum h_i x_i + sum_{i<j} J_ij x_i x_j for bitstring bits."""
    x = bits
    val = float(qubo.offset_base)
    for i, h in qubo.linear_base.items():
        val += float(h) * float(x[i])
    for (i, j), J in qubo.quadratic_base.items():
        val += float(J) * float(x[i]) * float(x[j])
    return val

def decode_from_qubo(qubo: Any, bits: np.ndarray) -> Dict[str, Any]:
    """Decode a QUBO bitstring, allowing non-VRP case studies to override it."""
    if hasattr(qubo, "decode_bits") and callable(qubo.decode_bits):
        return qubo.decode_bits(bits)
    x_assign = {(i, t, k): int(bits[idx]) for (i, t, k), idx in qubo.x_index.items()}
    y_assign = {(k, b): int(bits[idx]) for (k, b), idx in qubo.y_index.items()}
    return {"x": x_assign, "y": y_assign}
