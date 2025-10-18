from __future__ import annotations
import argparse
import os
import csv
from ant_colony import AntColonyAssignment, ACOParams
from ant_colony.utils import generate_matrix, greedy_baseline
from hungarian_manual import hungarian_solve
from hungarian_scipy import hungarian_scipy_solve


def rel_err(cost, opt):
    if opt is None or opt <= 0:
        return ""
    return (cost - opt) / opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mode", type=str, default="uniform")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time_limit", type=float, default=1.0)
    ap.add_argument("--ants", type=int, default=120)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--out_csv", type=str, default="compare_out/summary.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    C = generate_matrix(args.n, args.mode, args.seed)

    # ground truth via SciPy
    opt_cost = None
    try:
        sci = hungarian_scipy_solve(C);
        opt_cost = sci.cost
    except Exception as e:
        print("SciPy not available, skipping exact optimum.")

    # manual (time-bounded)
    hm = hungarian_solve(C, time_limit=args.time_limit)

    # ACO (time-bounded)
    aco = AntColonyAssignment(
        ACOParams(num_ants=args.ants, max_iters=args.iters, time_limit=args.time_limit, seed=args.seed,
                  local_search="2swap"))
    p, aco_cost, info = aco.solve(C)

    # Greedy
    _, gb_cost = greedy_baseline(C)

    rows = []

    def add(name, status, cost, elapsed, iterations=""):
        rows.append({"algo": name, "n": args.n, "mode": args.mode, "seed": args.seed, "time_limit": args.time_limit,
                     "status": status, "cost": cost, "opt_cost": opt_cost, "rel_error": rel_err(cost, opt_cost),
                     "elapsed": elapsed, "iterations": iterations})

    if opt_cost is not None: add("hungarian_scipy", "ok", sci.cost, sci.elapsed, "")
    add("hungarian_manual", hm.status, hm.cost, hm.elapsed, hm.iterations)
    add("aco", "ok", aco_cost, info["elapsed"], info["iterations"])
    add("greedy", "ok", gb_cost, 0.0, "")

    write_header = not os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header: w.writeheader()
        for r in rows: w.writerow(r)

    print(f"Wrote {args.out_csv}")
    for r in rows: print(r)


if __name__ == "__main__":
    main()
