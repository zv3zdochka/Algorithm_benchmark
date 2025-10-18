from __future__ import annotations
import argparse, numpy as np
from .hungarian import hungarian_solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--mode", type=str, default="uniform")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time_limit", type=float, default=None)
    args = ap.parse_args()
    if args.csv:
        C = np.loadtxt(args.csv, delimiter=",", dtype=float)
    else:
        rng = np.random.default_rng(args.seed)
        C = rng.random((args.n, args.n))
    res = hungarian_solve(C, time_limit=args.time_limit)
    print(f"status={res.status} n={C.shape[0]} cost={res.cost:.6f} elapsed={res.elapsed:.3f}s iters={res.iterations}")
    if C.shape[0] <= 20:
        print("perm:", res.perm.tolist())


if __name__ == "__main__":
    main()
