from __future__ import annotations
import argparse, os, time, csv
from typing import Optional, List, Dict
import numpy as np


def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    return plt


from ant_colony.aco import AntColonyAssignment, ACOParams
from ant_colony.utils import generate_matrix, greedy_baseline, save_history_csv
from hungarian_manual.hungarian import hungarian_solve
from hungarian_scipy.hungarian_lib import hungarian_scipy_solve

ALGOS = ["aco", "hungarian_manual", "hungarian_scipy", "greedy", "all"]

# Человечные подписи алгоритмов на русском
ALGO_LABELS_RU = {
    "aco": "Муравьиный алг-м",
    "hungarian_manual": "Венгерский",
    "hungarian_scipy": "Венгерский SciPy",
    "greedy": "Жадный",
}


def display_algo(name: str) -> str:
    return ALGO_LABELS_RU.get(name, name)


def metric_label(metric: str) -> str:
    """Человечные подписи метрик на русском."""
    m = metric.lower()
    if m == "elapsed":
        return "Время выполнения, с"
    if m == "rel_error":
        return "Относительная ошибка (доля)\n(отн. библиотечного Венгерского)"
    if m == "cost":
        return "Стоимость решения"
    if m == "iterations":
        return "Число итераций"
    return metric.replace("_", " ").title()


def load_matrix_from_csv(path: str) -> np.ndarray:
    import csv
    vals: List[List[float]] = []
    with open(path, "r", newline="") as f:
        for row in csv.reader(f):
            if not row: continue
            vals.append([float(x) for x in row])
    C = np.array(vals, dtype=float)
    assert C.ndim == 2 and C.shape[0] == C.shape[1], "CSV must be square matrix"
    return C


def ensure_outdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def run_aco(C: np.ndarray, args) -> Dict[str, object]:
    params = ACOParams(
        num_ants=args.ants,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q0=args.q0,
        tau0=None,
        Q=args.Q,
        local_search=args.local_search,
        ls_iters=args.ls_iters,
        max_iters=args.iters,
        time_limit=args.time_limit,
        seed=args.seed,
        verbose=args.verbose,
    )
    aco = AntColonyAssignment(params)
    t0 = time.time()
    p, cost, info = aco.solve(C)
    elapsed = time.time() - t0
    if args.history_csv:
        save_history_csv(info.get("history", []), args.history_csv)
    return {"algo": "aco", "perm": p, "cost": float(cost), "elapsed": elapsed, "status": "ok",
            "iterations": info.get("iterations", ""), "history": info.get("history", [])}


def run_hungarian_manual(C: np.ndarray, args) -> Dict[str, object]:
    res = hungarian_solve(C, time_limit=args.time_limit)
    return {"algo": "hungarian_manual", "perm": res.perm, "cost": float(res.cost),
            "elapsed": float(res.elapsed), "status": res.status, "iterations": res.iterations}


def run_hungarian_scipy(C: np.ndarray, args) -> Dict[str, object]:
    res = hungarian_scipy_solve(C)
    return {"algo": "hungarian_scipy", "perm": res.perm, "cost": float(res.cost),
            "elapsed": float(res.elapsed), "status": "ok", "iterations": ""}


def run_greedy(C: np.ndarray, args) -> Dict[str, object]:
    t0 = time.time()
    p, cost = greedy_baseline(C)
    elapsed = time.time() - t0
    return {"algo": "greedy", "perm": p, "cost": float(cost), "elapsed": elapsed, "status": "ok", "iterations": ""}


def rel_error(cost: float, opt_cost: Optional[float]) -> str:
    if opt_cost is None or opt_cost <= 0:
        return ""
    return (cost - opt_cost) / opt_cost


def plot_convergence(history: List[Dict[str, float]], out_png: str, title: str):
    if not history:
        return
    plt = _lazy_import_matplotlib()
    xs = [h["iter"] for h in history]
    best = [h["best_cost"] for h in history]
    mean = [h["mean_cost"] for h in history]
    plt.figure()
    plt.plot(xs, best, label="Лучшее")
    plt.plot(xs, mean, label="Среднее")
    plt.xlabel("Итерация")
    plt.ylabel("Стоимость решения")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_bars(per_n_rows: List[Dict[str, object]], out_png: str, metric: str, title: str):
    if not per_n_rows:
        return
    plt = _lazy_import_matplotlib()
    from collections import defaultdict
    sums, counts = defaultdict(float), defaultdict(int)
    for r in per_n_rows:
        key = r["algo"]
        v = r.get(metric, None)
        if v is None or v == "":
            continue
        v = float(v)
        sums[key] += v
        counts[key] += 1
    algos = sorted(sums.keys())
    vals = [sums[a] / max(1, counts[a]) for a in algos]
    algos_display = [display_algo(a) for a in algos]
    plt.figure()
    plt.bar(algos_display, vals)
    plt.ylabel(metric_label(metric))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_lines(summary_csv: str, out_png: str, metric: str, title: str):
    import csv
    rows = []
    with open(summary_csv, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows: return
    plt = _lazy_import_matplotlib()
    algos = sorted(set(r["algo"] for r in rows))

    plt.figure()
    for a in algos:
        xs, ys = [], []
        for r in rows:
            if r["algo"] != a: continue
            n = int(r["n"])
            v = r.get(metric, "")
            if v == "": continue
            xs.append(n)
            ys.append(float(v))
        if xs:
            pairs = sorted(zip(xs, ys), key=lambda t: t[0])
            xs = [p[0] for p in pairs]
            ys = [p[1] for p in pairs]
            plt.plot(xs, ys, label=display_algo(a))
    plt.xlabel("Размер матрицы, n")
    plt.ylabel(metric_label(metric))
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_rows(rows: List[Dict[str, object]], out_csv: str):
    if not rows:
        return
    keys = ["algo", "n", "mode", "seed", "time_limit", "status", "cost", "opt_cost", "rel_error", "elapsed",
            "iterations"]
    write_header = not os.path.exists(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header: w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in keys})


def run_once(args, n: int, C: Optional[np.ndarray]) -> List[Dict[str, object]]:
    if C is None:
        if args.csv:
            C = load_matrix_from_csv(args.csv)
        else:
            C = generate_matrix(n=n, mode=args.mode, seed=args.seed)
    opt = run_hungarian_scipy(C, args)
    opt_cost = opt["cost"]
    rows: List[Dict[str, object]] = []

    def add_row(res: Dict[str, object]):
        rows.append({
            "algo": res["algo"],
            "n": n,
            "mode": args.mode if not args.csv else "csv",
            "seed": args.seed,
            "time_limit": args.time_limit if res["algo"] in ("aco", "hungarian_manual") else "",
            "status": res.get("status", "ok"),
            "cost": res["cost"],
            "opt_cost": opt_cost,
            "rel_error": rel_error(res["cost"], opt_cost),
            "elapsed": res["elapsed"],
            "iterations": res.get("iterations", ""),
        })

    if args.algo in ("aco", "all"):
        res = run_aco(C, args)
        add_row(res)
        if args.outdir:
            plot_convergence(
                res.get("history", []),
                os.path.join(args.outdir, f"convergence_aco_n{n}.png"),
                f"Сходимость ACO (n={n})"
            )
            if args.history_csv:
                # сохранить историю отдельно, если просили
                pass
    if args.algo in ("hungarian_manual", "all"):
        res = run_hungarian_manual(C, args)
        add_row(res)
    if args.algo in ("hungarian_scipy", "all"):
        res = run_hungarian_scipy(C, args)
        add_row(res)
    if args.algo in ("greedy", "all"):
        res = run_greedy(C, args)
        add_row(res)

    return rows


def main():
    p = argparse.ArgumentParser(description="Единый бенчмарк: один алгоритм или все сразу.")
    p.add_argument("--algo", choices=ALGOS, default="all")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--sizes", type=str, default="")
    p.add_argument("--mode", type=str, default="uniform")
    p.add_argument("--csv", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time_limit", type=float, default=None)

    # ACO
    p.add_argument("--ants", type=int, default=80)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--rho", type=float, default=0.10)
    p.add_argument("--q0", type=float, default=0.20)
    p.add_argument("--Q", type=float, default=1.0)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--ls_iters", type=int, default=200)
    p.add_argument("--local_search", type=str, default="2swap")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--outdir", type=str, default="out")
    p.add_argument("--out_csv", type=str, default="out/summary.csv")
    p.add_argument("--history_csv", type=str, default="")

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.csv:
        C = load_matrix_from_csv(args.csv)
        sizes = [C.shape[0]]
    elif args.sizes:
        sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
        C = None
    else:
        sizes = [args.n]
        C = None

    all_rows: List[Dict[str, object]] = []
    for n in sizes:
        rows = run_once(args, n, C if (args.csv and C is not None and C.shape[0] == n) else None)
        # Пер-n графики (столбики)
        save_rows(rows, args.out_csv)

        per_n_png_elapsed = os.path.join(args.outdir, f"bars_elapsed_n{n}.png")
        try:
            from matplotlib import pyplot as plt  # проверка наличия matplotlib

            def _plot_bars(rows_local, out_png, metric, title):
                import matplotlib.pyplot as plt
                algos_local = [display_algo(r["algo"]) for r in rows_local]
                vals_local = [float(r.get(metric, 0) or 0) if r.get(metric, "") != "" else 0.0 for r in rows_local]
                plt.figure()
                plt.bar(algos_local, vals_local)
                plt.ylabel(metric_label(metric))
                plt.title(title)
                plt.tight_layout()
                plt.savefig(out_png)
                plt.close()

            _plot_bars(rows, per_n_png_elapsed, "elapsed", f"Время выполнения по алгоритмам (n={n})")
            _plot_bars(rows, os.path.join(args.outdir, f"bars_rel_error_n{n}.png"), "rel_error",
                       f"Относительная ошибка по алгоритмам (n={n})\n(относительно библиотечного Венгерского)")
        except Exception:
            pass

        all_rows.extend(rows)

    # Линейные графики по размерам
    if len(sizes) > 1 or args.algo == "all":
        try:
            import matplotlib.pyplot as plt
            import csv as _csv
            dat = []
            with open(args.out_csv, "r", newline="") as f:
                for r in _csv.DictReader(f):
                    dat.append(r)

            def _plot_lines(metric, out_png, title):
                algos = sorted(set(r["algo"] for r in dat))
                plt.figure()
                for a in algos:
                    xs, ys = [], []
                    for r in dat:
                        if r["algo"] != a: continue
                        v = r.get(metric, "")
                        if v == "": continue
                        xs.append(int(r["n"]))
                        ys.append(float(v))
                    if xs:
                        pairs = sorted(zip(xs, ys))
                        xs = [p[0] for p in pairs]
                        ys = [p[1] for p in pairs]
                        plt.plot(xs, ys, label=display_algo(a))
                plt.xlabel("Размер матрицы, n")
                plt.ylabel(metric_label(metric))
                plt.title(title)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_png)
                plt.close()

            _plot_lines("elapsed", os.path.join(args.outdir, "lines_elapsed_vs_n.png"),
                        "Зависимость времени выполнения от размера матрицы")
            _plot_lines("rel_error", os.path.join(args.outdir, "lines_rel_error_vs_n.png"),
                        "Зависимость относительной ошибки от размера матрицы\n(относительно библиотечного Венгерского)")
        except Exception:
            pass

    for r in all_rows:
        print(r)


if __name__ == "__main__":
    main()
