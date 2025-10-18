from __future__ import annotations
import argparse
import os
from ant_colony.aco import AntColonyAssignment, ACOParams
from ant_colony.utils import generate_matrix, save_history_csv, greedy_baseline


def run_one(n, mode, seed, time_limit, iters, ants):
    C = generate_matrix(n, mode=mode, seed=seed)
    params = ACOParams(num_ants=ants, max_iters=iters, time_limit=time_limit, seed=seed, local_search="2swap",
                       verbose=False)
    aco = AntColonyAssignment(params)
    p, cost, info = aco.solve(C)
    gb_p, gb_cost = greedy_baseline(C)
    return C, p, cost, gb_cost, info


def plot_history(history, out_png, title):
    it = [h["iter"] for h in history];
    best = [h["best_cost"] for h in history];
    mean = [h["mean_cost"] for h in history]
    import matplotlib.pyplot as plt
    plt.figure();
    plt.plot(it, best, label="Best", marker="o");
    plt.plot(it, mean, label="Mean", marker="o")
    plt.xlabel("Iteration");
    plt.ylabel("Cost");
    plt.title(title);
    plt.legend();
    plt.tight_layout();
    plt.savefig(out_png, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80);
    ap.add_argument("--mode", type=str, default="uniform")
    ap.add_argument("--seed", type=int, default=42);
    ap.add_argument("--time_limit", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=500);
    ap.add_argument("--ants", type=int, default=60)
    ap.add_argument("--outdir", type=str, default="bench_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    C, p, cost, gb_cost, info = run_one(args.n, args.mode, args.seed, args.time_limit, args.iters, args.ants)
    csv_path = os.path.join(args.outdir, "history.csv");
    png_path = os.path.join(args.outdir, "convergence.png")
    from ant_colony.utils import save_history_csv;
    save_history_csv(info["history"], csv_path)
    plot_history(info["history"], png_path, f"ACO n={args.n} mode={args.mode}")
    print(f"Best cost: {cost:.6f} | Greedy: {gb_cost:.6f}");
    print(f"Saved: {csv_path}");
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
