from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np, time

@dataclass
class HungarianResult:
    perm: np.ndarray
    cost: float
    status: str
    elapsed: float
    iterations: int

def _complete_greedy(C0: np.ndarray, starred: np.ndarray) -> np.ndarray:
    n = C0.shape[0]
    perm = -np.ones(n, dtype=int)
    used_cols = set()

    for i in range(n):
        js = np.where(starred[i])[0]
        if js.size > 0:
            j = int(js[0])
            perm[i] = j
            used_cols.add(j)

    for i in range(n):
        if perm[i] != -1:
            continue
        candidates = [j for j in range(n) if j not in used_cols]
        if not candidates:
            candidates = list(range(n))
        jbest = min(candidates, key=lambda j: C0[i, j])
        perm[i] = jbest
        used_cols.add(jbest)
    return perm

def hungarian_solve(C: np.ndarray, time_limit: Optional[float] = None) -> HungarianResult:
    t0 = time.time()
    deadline = (t0 + time_limit) if time_limit is not None else None
    eps = 1e-12

    C0 = np.array(C, dtype=float)
    C  = C0.copy()
    n  = C.shape[0]
    assert C.ndim == 2 and n == C.shape[1], "C must be square"

    C -= C.min(axis=1, keepdims=True)
    C -= C.min(axis=0, keepdims=True)

    starred = np.zeros((n, n), dtype=bool)
    primed  = np.zeros((n, n), dtype=bool)
    row_cov = np.zeros(n, dtype=bool)
    col_cov = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if abs(C[i, j]) <= eps and not row_cov[i] and not col_cov[j]:
                starred[i, j] = True
                row_cov[i] = True
                col_cov[j] = True
                break
    row_cov[:] = False
    col_cov[:] = False

    iterations = 0
    while True:
        iterations += 1
        if deadline and time.time() >= deadline:
            perm = _complete_greedy(C0, starred)
            true_cost = float(C0[np.arange(n), perm].sum())
            return HungarianResult(
                perm=perm, cost=true_cost, status="timeout",
                elapsed=(time.time() - t0), iterations=iterations
            )

        col_cov[:] = np.any(starred, axis=0)
        if col_cov.sum() == n:
            perm = np.argmax(starred, axis=1)
            true_cost = float(C0[np.arange(n), perm].sum())
            return HungarianResult(
                perm=perm, cost=true_cost, status="ok",
                elapsed=(time.time() - t0), iterations=iterations
            )

        while True:
            if deadline and time.time() >= deadline:
                perm = _complete_greedy(C0, starred)
                true_cost = float(C0[np.arange(n), perm].sum())
                return HungarianResult(
                    perm=perm, cost=true_cost, status="timeout",
                    elapsed=(time.time() - t0), iterations=iterations
                )

            found = False
            for i in range(n):
                if row_cov[i]:
                    continue
                for j in range(n):
                    if col_cov[j]:
                        continue
                    if abs(C[i, j]) <= eps and not starred[i, j]:
                        primed[i, j] = True
                        found = True
                        star_in_row = np.where(starred[i])[0]
                        if star_in_row.size > 0:
                            row_cov[i] = True
                            col_cov[star_in_row[0]] = False
                        else:
                            path = [(i, j)]
                            while True:
                                if deadline and time.time() >= deadline:
                                    perm = _complete_greedy(C0, starred)
                                    true_cost = float(C0[np.arange(n), perm].sum())
                                    return HungarianResult(
                                        perm=perm, cost=true_cost, status="timeout",
                                        elapsed=(time.time() - t0), iterations=iterations
                                    )
                                r = np.where(starred[:, path[-1][1]])[0]
                                if r.size == 0:
                                    break
                                r = int(r[0])
                                path.append((r, path[-1][1]))
                                cands = np.where(primed[r])[0]
                                if len(path) >= 2:
                                    cands = cands[cands != path[-2][1]]
                                if cands.size == 0:
                                    break
                                c = int(cands[0])
                                path.append((r, c))
                            for (r, c) in path:
                                if primed[r, c]:
                                    starred[r, c] = True
                                    primed[r, c] = False
                                else:
                                    starred[r, c] = False
                            row_cov[:] = False
                            col_cov[:] = False
                            primed[:] = False
                            break
                        break
                if found:
                    break

            if not found:
                mask_rows = ~row_cov
                mask_cols = ~col_cov
                if np.any(mask_rows) and np.any(mask_cols):
                    sub = C[np.ix_(mask_rows, mask_cols)]
                    m = np.min(sub)
                else:
                    m = 0.0
                if m > eps:
                    C[mask_rows] -= m
                    C[:, col_cov] += m
            else:
                break
