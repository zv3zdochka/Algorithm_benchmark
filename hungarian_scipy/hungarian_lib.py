from __future__ import annotations
from dataclasses import dataclass
import numpy as np, time


@dataclass
class SciPyHungarianResult:
    perm: np.ndarray
    cost: float
    elapsed: float


def hungarian_scipy_solve(C: np.ndarray) -> SciPyHungarianResult:
    t0 = time.time()
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:
        raise RuntimeError("SciPy is required (pip install scipy)") from e
    C = np.array(C, dtype=float)
    r, c = linear_sum_assignment(C)
    perm = np.empty(C.shape[0], dtype=int);
    perm[r] = c
    cost = float(C[r, c].sum())
    return SciPyHungarianResult(perm=perm, cost=cost, elapsed=(time.time() - t0))
