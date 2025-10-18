from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, List
import itertools
import csv
import os


def brute_force_opt(C: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Exact optimum via brute-force permutations (ONLY for very small n <= 9).
    Returns best permutation and cost.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    assert n == C.shape[1]
    best_cost = float("inf")
    best_p = None
    for p in itertools.permutations(range(n)):
        cost = float(C[np.arange(n), list(p)].sum())
        if cost < best_cost:
            best_cost = cost
            best_p = np.array(p, dtype=int)
    return best_p, best_cost


def greedy_baseline(C: np.ndarray, k: int = 1) -> Tuple[np.ndarray, float]:
    """
    Simple greedy: repeatedly pick globally smallest unused cell (row, col not used).
    k allows keeping k candidates per row to slightly improve decisions; default 1.
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    rows_used = set()
    cols_used = set()
    assignment = [-1] * n
    # Flatten and sort
    flat = [(C[i, j], i, j) for i in range(n) for j in range(n)]
    flat.sort(key=lambda x: x[0])
    for _, i, j in flat:
        if i in rows_used or j in cols_used:
            continue
        assignment[i] = j
        rows_used.add(i)
        cols_used.add(j)
        if len(rows_used) == n:
            break
    p = np.array(assignment, dtype=int)
    cost = float(C[np.arange(n), p].sum())
    return p, cost


def generate_matrix(n: int, mode: str = "uniform", seed: int = 42, **kwargs) -> np.ndarray:
    """
    Generate synthetic cost matrices.
    Modes:
      - uniform: U[0,1)
      - normal: N(0,1) abs
      - diag_dom: diagonal dominant near-zero diagonal, larger off-diagonal
      - block: clusters of low-cost blocks
    """
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        C = rng.random((n, n))
    elif mode == "normal":
        C = np.abs(rng.normal(0, 1, (n, n)))
    elif mode == "diag_dom":
        C = 0.1 * rng.random((n, n)) + 1.0
        np.fill_diagonal(C, 0.01 * rng.random(n))
    elif mode == "block":
        C = 1.0 + rng.random((n, n))
        b = kwargs.get("blocks", 4)
        bs = n // b
        for bi in range(b):
            for bj in range(b):
                si = slice(bi * bs, min((bi + 1) * bs, n))
                sj = slice(bj * bs, min((bj + 1) * bs, n))
                C[si, sj] *= rng.uniform(0.1, 0.5)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return C


def save_history_csv(history: List[Dict[str, Any]], path: str) -> None:
    if not history:
        return
    keys = ["iter", "elapsed", "iter_best_cost", "best_cost", "mean_cost", "std_cost"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in keys})
