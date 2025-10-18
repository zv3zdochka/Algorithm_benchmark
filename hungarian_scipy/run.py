from __future__ import annotations
import argparse, numpy as np
from .hungarian_lib import hungarian_scipy_solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--mode", type=str, default="uniform")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.csv:
        C = np.loadtxt(args.csv, delimiter=",", dtype=float)
    else:
        rng = np.random.default_rng(args.seed);
        C = rng.random((args.n, args.n))
    res = hungarian_scipy_solve(C)
    print(f"status=ok n={C.shape[0]} cost={res.cost:.6f} elapsed={res.elapsed:.6f}s")
    if C.shape[0] <= 20: print("perm:", res.perm.tolist())


if __name__ == "__main__":
    main()
