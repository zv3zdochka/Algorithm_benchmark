from __future__ import annotations

import argparse
import numpy as np
from .aco import AntColonyAssignment, ACOParams
from .utils import generate_matrix, save_history_csv


def load_matrix(args) -> np.ndarray:
    if args.csv:
        return np.loadtxt(args.csv, delimiter=",", dtype=float)
    elif args.npy:
        return np.load(args.npy)
    else:
        return generate_matrix(args.n, mode=args.mode, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description="ACO for Assignment Problem")
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV matrix")
    parser.add_argument("--npy", type=str, default=None, help="Path to .npy matrix")
    parser.add_argument("--n", type=int, default=40, help="Size for generated matrix")
    parser.add_argument("--mode", type=str, default="uniform", help="Matrix gen mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ants", type=int, default=40, help="Number of ants")
    parser.add_argument("--iters", type=int, default=500, help="Max ACO iterations")
    parser.add_argument("--time_limit", type=float, default=None, help="Seconds budget")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--rho", type=float, default=0.10)
    parser.add_argument("--q0", type=float, default=0.20)
    parser.add_argument("--tau0", type=float, default=None)
    parser.add_argument("--Q", type=float, default=1.0)
    parser.add_argument("--local_search", type=str, default="2swap", choices=["", "2swap"])
    parser.add_argument("--ls_iters", type=int, default=200)
    parser.add_argument("--history_csv", type=str, default=None, help="Where to save history CSV")
    args = parser.parse_args()

    C = load_matrix(args)
    params = ACOParams(
        num_ants=args.ants,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q0=args.q0,
        tau0=args.tau0,
        Q=args.Q,
        local_search=args.local_search,
        ls_iters=args.ls_iters,
        max_iters=args.iters,
        time_limit=args.time_limit,
        seed=args.seed,
        verbose=True,
    )
    aco = AntColonyAssignment(params)
    p, cost, info = aco.solve(C)

    print("=== ACO Result ===")
    print(f"n={C.shape[0]} best_cost={cost:.6f} iters={info['iterations']} elapsed={info['elapsed']:.3f}s")
    if C.shape[0] <= 20:
        print("perm:", p.tolist())
    if args.history_csv:
        save_history_csv(info["history"], args.history_csv)
        print(f"History saved to: {args.history_csv}")


if __name__ == "__main__":
    main()
