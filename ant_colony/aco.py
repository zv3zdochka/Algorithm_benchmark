from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import time
import random
import numpy as np


@dataclass
class ACOParams:
    num_ants: int = 40
    alpha: float = 1.0
    beta: float = 3.0
    rho: float = 0.10
    q0: float = 0.20
    tau0: Optional[float] = None
    Q: float = 1.0
    local_search: str = "2swap"
    ls_iters: int = 200
    max_iters: int = 1000
    time_limit: Optional[float] = None
    seed: Optional[int] = None
    verbose: bool = False


class AntColonyAssignment:

    def __init__(self, params: Optional[ACOParams] = None):
        self.params = params or ACOParams()
        self.rng = random.Random(self.params.seed)
        np.random.seed(self.params.seed if self.params.seed is not None else None)

    def _normalize_costs(self, C: np.ndarray) -> np.ndarray:
        C = np.asarray(C, dtype=float)
        mn, mx = np.min(C), np.max(C)
        if mx - mn < 1e-12:
            return np.ones_like(C)
        return (C - mn) / (mx - mn + 1e-12)

    def _heuristic(self, C_norm: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return 1.0 / (C_norm + eps)

    def _cost_of_perm(self, C: np.ndarray, p: np.ndarray) -> float:
        return float(C[np.arange(C.shape[0]), p].sum())

    def _greedy_baseline(self, C: np.ndarray) -> Tuple[np.ndarray, float]:
        n = C.shape[0]
        rows_used, cols_used = set(), set()
        assignment = [-1] * n
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
        return p, self._cost_of_perm(C, p)

    def _greedy_complete_remaining(self, C: np.ndarray, current_p: np.ndarray,
                                   filled_mask: np.ndarray, available_cols: List[int]) -> None:
        n = C.shape[0]
        candidates = []
        remaining_rows = [i for i in range(n) if not filled_mask[i]]
        for i in remaining_rows:
            for j in available_cols:
                candidates.append((C[i, j], i, j))
        candidates.sort(key=lambda x: x[0])
        used_cols = set()
        for _, i, j in candidates:
            if filled_mask[i] or j in used_cols:
                continue
            current_p[i] = j
            filled_mask[i] = True
            used_cols.add(j)
            if len(used_cols) == len(available_cols):
                break

    def _construct_solution(self, C: np.ndarray, tau: np.ndarray, eta: np.ndarray,
                            deadline: Optional[float]) -> np.ndarray:
        n = tau.shape[0]
        available = list(range(n))
        p = -np.ones(n, dtype=int)
        filled = np.zeros(n, dtype=bool)

        rows = list(range(n))
        self.rng.shuffle(rows)
        alpha, beta, q0 = self.params.alpha, self.params.beta, self.params.q0

        for i in rows:
            if deadline is not None and time.time() >= deadline:
                self._greedy_complete_remaining(C, p, filled, available)
                if np.any(~filled):
                    p, _ = self._greedy_baseline(C)
                return p

            vals = []
            for j in available:
                vals.append((j, (tau[i, j] ** alpha) * (eta[i, j] ** beta)))
            total = sum(v for _, v in vals) + 1e-32

            if self.rng.random() < q0:
                j = max(vals, key=lambda x: x[1])[0]
            else:
                r = self.rng.random() * total
                acc = 0.0
                j = available[-1]
                for jj, v in vals:
                    acc += v
                    if acc >= r:
                        j = jj
                        break

            p[i] = j
            filled[i] = True
            available.remove(j)

        return p

    def _two_swap_local_search(self, C: np.ndarray, p: np.ndarray,
                               deadline: Optional[float]) -> Tuple[np.ndarray, float]:
        best_p = p.copy()
        best_cost = self._cost_of_perm(C, best_p)
        n = len(p)
        for _ in range(self.params.ls_iters):
            if deadline is not None and time.time() >= deadline:
                break
            i, k = self.rng.randrange(n), self.rng.randrange(n)
            if i == k:
                continue
            j_i, j_k = best_p[i], best_p[k]
            delta = (C[i, j_k] + C[k, j_i]) - (C[i, j_i] + C[k, j_k])
            if delta < -1e-15:
                best_p[i], best_p[k] = j_k, j_i
                best_cost += delta
        return best_p, best_cost

    def solve(self, C: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        C = np.asarray(C, dtype=float)
        assert C.ndim == 2 and C.shape[0] == C.shape[1], "C must be square n x n matrix"
        n = C.shape[0]

        C_norm = self._normalize_costs(C)
        eta = self._heuristic(C_norm)

        if self.params.tau0 is not None:
            tau0 = float(self.params.tau0)
        else:
            tau0 = 1.0 / (np.mean(C_norm) + 1e-9)
        tau = np.full_like(C, tau0, dtype=float)

        best_p: Optional[np.ndarray] = None
        best_cost = float('inf')

        history: List[Dict[str, Any]] = []
        t0 = time.time()
        deadline = (t0 + self.params.time_limit) if self.params.time_limit is not None else None

        def time_up() -> bool:
            return (deadline is not None) and (time.time() >= deadline)

        it = 0
        while it < self.params.max_iters and not time_up():
            costs: List[float] = []
            sols: List[np.ndarray] = []

            for _ in range(self.params.num_ants):
                if time_up():
                    break
                p = self._construct_solution(C, tau, eta, deadline)

                if self.params.local_search == "2swap" and not time_up():
                    p, cost = self._two_swap_local_search(C, p, deadline)
                else:
                    cost = self._cost_of_perm(C, p)

                costs.append(cost)
                sols.append(p)

            if not costs:
                break

            idx = int(np.argmin(costs))
            it_best_cost = float(costs[idx])
            it_best_p = sols[idx]

            if it_best_cost < best_cost - 1e-15:
                best_cost = it_best_cost
                best_p = it_best_p.copy()

            rho = self.params.rho
            tau *= (1.0 - rho)
            if best_p is None:
                best_p, best_cost = self._greedy_baseline(C)
            deposit = self.params.Q / (best_cost + 1e-12)
            for i in range(n):
                tau[i, best_p[i]] += deposit
            tau = np.clip(tau, 1e-12, 1e12)

            it += 1
            elapsed = time.time() - t0
            history.append({
                "iter": it,
                "elapsed": elapsed,
                "iter_best_cost": it_best_cost,
                "best_cost": best_cost,
                "mean_cost": float(np.mean(costs)),
                "std_cost": float(np.std(costs)),
            })

            if self.params.verbose and (it % 10 == 0):
                print(f"[ACO] iter={it} best={best_cost:.6f} it_best={it_best_cost:.6f} elapsed={elapsed:.3f}s")

        if best_p is None:
            best_p, best_cost = self._greedy_baseline(C)

        info: Dict[str, Any] = {
            "history": history,
            "tau": tau,
            "params": self.params.__dict__,
            "elapsed": (time.time() - t0),
            "iterations": it,
        }
        return best_p, best_cost, info
