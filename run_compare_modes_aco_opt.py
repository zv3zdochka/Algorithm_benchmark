#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_compare_modes.py

Сравнение алгоритмов назначения:
  - Муравьиный алгоритм (ACO)
  - Венгерский (собственная реализация)
  - Венгерский (библиотечный, SciPy)
  - Жадный

Два режима:
  - Режим A: «достаточно времени» (--time_A)
  - Режим B: «жёсткий таймаут» (--time_B)

ВАЖНО:
  1) В качестве «правильного ответа» используем БИБЛИОТЕЧНЫЙ венгерский БЕЗ лимита.
     Сначала считаем истинный opt_cost для каждой пары (n, seed),
     затем пересчитываем rel_error всех строк как (cost / opt_true - 1).
     Поэтому библиотечный венгерский на всех графиках имеет rel_error = 0 (базовая линия).
  2) ACO запускается с фиксированными пер-размерными параметрами (таблица ниже).
     Мы НЕ переопределяем экспериментальный seed (0..repeats-1). Переопределение seed из
     таблицы ACO УБРАНО, чтобы инстанс матрицы совпадал у всех алгоритмов.
  3) Подписи графиков — на русском. Добавлены пояснения и подписи осей.
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Шрифты с кириллицей и минусы в осях
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

PY = sys.executable

# --- размеры по умолчанию ---
DEFAULT_SIZES = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000, 3000, 4000, 5000]

# --- «человечные» подписи алгоритмов на русском ---
ALGO_LABELS_RU = {
    "aco": "Муравьиный алгоритм (ACO)",
    "hungarian_manual": "Венгерский",
    "hungarian_scipy": "Венгерский SciPy",
    "greedy": "Жадный",
}

# --- таблица параметров ACO по размерам (из твоего файла) ---
ACO_PARAMS_TABLE = [
    dict(n=25, ants=40, alpha=1.1, beta=4.1, rho=0.32, q0=0.32, local_search="2swap", ls_iters=203, max_iters=200),
    dict(n=50, ants=27, alpha=1.4, beta=5.5, rho=0.26, q0=0.89, local_search="2swap", ls_iters=260, max_iters=200),
    dict(n=75, ants=32, alpha=1.0, beta=5.3, rho=0.40, q0=0.28, local_search="2swap", ls_iters=87, max_iters=200),
    dict(n=100, ants=24, alpha=1.4, beta=5.1, rho=0.35, q0=0.39, local_search="2swap", ls_iters=115, max_iters=200),
    dict(n=150, ants=38, alpha=1.2, beta=5.7, rho=0.07, q0=0.54, local_search="2swap", ls_iters=55, max_iters=150),
    dict(n=200, ants=39, alpha=1.4, beta=5.1, rho=0.31, q0=0.51, local_search="2swap", ls_iters=173, max_iters=150),
    dict(n=300, ants=20, alpha=1.2, beta=4.3, rho=0.15, q0=0.81, local_search="2swap", ls_iters=191, max_iters=150),
    dict(n=400, ants=23, alpha=1.4, beta=5.1, rho=0.33, q0=0.66, local_search="2swap", ls_iters=142, max_iters=150),
    dict(n=600, ants=32, alpha=0.9, beta=2.6, rho=0.34, q0=0.78, local_search="2swap", ls_iters=40, max_iters=120),
    dict(n=800, ants=43, alpha=0.8, beta=5.3, rho=0.19, q0=0.77, local_search="", ls_iters=0, max_iters=120),
    dict(n=1200, ants=16, alpha=0.9, beta=4.8, rho=0.18, q0=0.36, local_search="2swap", ls_iters=38, max_iters=120),
    dict(n=2000, ants=119, alpha=1.1, beta=5.5, rho=0.37, q0=0.21, local_search="2swap", ls_iters=54, max_iters=100),
    dict(n=3000, ants=105, alpha=1.1, beta=2.2, rho=0.06, q0=0.70, local_search="2swap", ls_iters=75, max_iters=100),
    dict(n=4000, ants=57, alpha=1.2, beta=5.6, rho=0.40, q0=0.17, local_search="", ls_iters=0, max_iters=100),
]
_ACO_BY_N: Dict[int, Dict[str, Any]] = {row["n"]: row for row in ACO_PARAMS_TABLE}


def get_aco_params_for_n(n: int) -> Dict[str, Any]:
    """Возвращает параметры ACO для данного n. Если точного нет — ближайший меньший (иначе максимальный доступный)."""
    if n in _ACO_BY_N:
        return dict(_ACO_BY_N[n])
    candidates = [k for k in _ACO_BY_N if k <= n]
    if candidates:
        use = max(candidates)
        print(f"[WARN] Нет параметров ACO для n={n}; использую ближайший n={use}")
        return dict(_ACO_BY_N[use])
    use = min(_ACO_BY_N)
    print(f"[WARN] Нет параметров ACO для n={n}; использую минимальный n={use}")
    return dict(_ACO_BY_N[use])


# ---------------- запуск «истины» и бенчмарка ----------------

def _run_unlimited_optimal_cost(n: int, seed: int, mode: str = "uniform") -> Optional[float]:
    """
    Вычисляем «истинный оптимум» библиотечным Венгерским БЕЗ лимита.
    Ожидается, что модуль печатает что-то со строкой 'cost=' в stdout.
    """
    cmd = [PY, "-m", "hungarian_scipy.run", "--n", str(n), "--seed", str(seed), "--mode", mode]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        out = proc.stdout or ""
        val = None
        for tok in out.replace(",", " ").split():
            if tok.startswith("cost="):
                try:
                    val = float(tok.split("=", 1)[1])
                    break
                except Exception:
                    pass
        if val is None:
            print(f"[ERROR] Не удалось распарсить opt_cost для n={n}, seed={seed}. Вывод:\n{out}")
        return val
    except Exception as e:
        print(f"[ERROR] Провал запуска hungarian_scipy.run для n={n}, seed={seed}: {e}")
        return None


def _append_aco_overrides(cmd: List[str], aco: Dict[str, Any]) -> None:
    """
    Добавляем флаги ACO в команду benchmark_unified.
    ВАЖНО: НЕ переопределяем --seed (чтобы совпадали инстансы у всех алгоритмов)!
    Подстрой имена флагов здесь, если в твоём benchmark_unified они другие.
    """
    ls = aco.get("local_search") or ""
    ls_flag = ls if ls else "none"
    cmd += [
        "--ants", str(aco.get("ants", 40)),
        "--alpha", str(aco.get("alpha", 1.0)),
        "--beta", str(aco.get("beta", 3.0)),
        "--rho", str(aco.get("rho", 0.1)),
        "--q0", str(aco.get("q0", 0.2)),
        "--local_search", ls_flag,
        "--ls_iters", str(aco.get("ls_iters", 200)),
        "--iters", str(aco.get("max_iters", 1000)),
    ]


def _run_benchmark_unified(n: int, seed: int, timelimit: float, outdir: str,
                           mode: str, aco_params: Dict[str, Any], verbose: bool = False) -> int:
    """Запуск единого бенча (все алгоритмы) с заданным лимитом времени и конкретными ACO-параметрами."""
    cmd = [
        PY, "-m", "bench.benchmark_unified",
        "--algo", "all",
        "--n", str(n),
        "--mode", mode,
        "--seed", str(seed),  # <- seed эксперимента, общий для всех алгоритмов
        "--time_limit", str(timelimit),
        "--outdir", outdir,
        "--out_csv", os.path.join(outdir, "summary.csv"),
    ]
    _append_aco_overrides(cmd, aco_params)
    if verbose:
        print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if verbose or proc.returncode != 0:
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"[WARN] benchmark_unified вернул код {proc.returncode} (n={n}, seed={seed})")
    return proc.returncode


# ---------------- перерасчёт метрик и графики ----------------

def _recompute_errors_with_true_opt(summary_csv: str, opt_map: Dict[Tuple[int, int], float]) -> pd.DataFrame:
    """Пересчитываем rel_error относительно «истинного» оптимума. Старые поля сохраняем как *_bench."""
    df = pd.read_csv(summary_csv)

    # типы
    for c in ("n", "seed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("cost", "opt_cost", "rel_error", "elapsed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # сохраняем как было
    if "opt_cost" in df.columns:
        df["opt_cost_bench"] = df["opt_cost"]
    if "rel_error" in df.columns:
        df["rel_error_bench"] = df["rel_error"]

    # новая опора — истинный оптимум
    true_opt = []
    for _, row in df.iterrows():
        key = (int(row["n"]), int(row.get("seed", 0)))
        true_opt.append(opt_map.get(key, np.nan))
    df["opt_cost"] = true_opt

    # новый rel_error
    df = df.dropna(subset=["cost", "opt_cost"])
    # защита от деления на ноль
    df = df[df["opt_cost"] != 0]
    df["rel_error"] = (df["cost"] / df["opt_cost"]) - 1.0
    df["rel_error_ref"] = "unlimited_hungarian"

    # «человечные» подписи алгоритмов
    df["algo_display"] = df["algo"].map(ALGO_LABELS_RU).fillna(df["algo"])

    df.to_csv(summary_csv, index=False)
    return df


def _safe_legend():
    """Вызывает legend(), только если есть что легендить."""
    handles, labels = plt.gca().get_legend_handles_labels()
    if len([l for l in labels if l and not l.startswith("_")]) > 0:
        plt.legend()


def _save_agg_and_plots(df: pd.DataFrame, outdir: str, title_suffix: str = "") -> None:
    """Агрегаты и подробные графики с подписями на русском."""
    df = df.copy()
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["rel_error"] = pd.to_numeric(df["rel_error"], errors="coerce")
    df = df.dropna(subset=["n", "elapsed", "rel_error", "algo_display"])

    # агрегаты
    agg = df.groupby(["algo_display", "n"]).agg(
        mean_rel_error=("rel_error", "mean"),
        std_rel_error=("rel_error", "std"),
        mean_elapsed=("elapsed", "mean"),
        std_elapsed=("elapsed", "std"),
        count=("rel_error", "count"),
        timeouts=("status", lambda s: (s == "timeout").sum() if s.notna().any() else 0),
    ).reset_index()
    agg_csv = os.path.join(outdir, f"agg_stats{('_' + title_suffix) if title_suffix else ''}.csv")
    agg.to_csv(agg_csv, index=False)
    print("Сохранены агрегаты:", agg_csv)

    # --- 1) Ошибка относительно библиотечного Венгерского ---
    plt.figure(figsize=(10, 6))
    plotted = False
    for algo, g in agg.groupby("algo_display"):
        if g.empty:
            continue
        xs = g["n"].values
        ys = 100.0 * g["mean_rel_error"].values
        yerr = 100.0 * g["std_rel_error"].fillna(0).values
        plt.errorbar(xs, ys, yerr=yerr, marker="o", label=algo)
        plotted = True
    if plotted:
        if agg["n"].max() / max(1, agg["n"].min()) > 20:
            plt.xscale("log")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.7)
        plt.xlabel("Размер матрицы, n")
        plt.ylabel("Относительная ошибка, %  (= 100 × (стоимость / оптимум − 1))")
        plt.title("Относительная ошибка относительно библиотечного Венгерского"
                  + (f" — {title_suffix}" if title_suffix else ""))
        plt.text(0.02, 0.02,
                 "Базовая линия y=0 — оптимум от библиотечного Венгерского (без лимита времени).",
                 transform=plt.gca().transAxes, fontsize=9, alpha=0.8)
        plt.grid(True, which="both", ls="--", alpha=0.4)
        _safe_legend()
    out_png = os.path.join(outdir, f"rel_error_vs_n{('_' + title_suffix) if title_suffix else ''}.png")
    plt.tight_layout();
    plt.savefig(out_png);
    plt.close()
    print("Сохранён график:", out_png)

    # --- 2) Время работы ---
    eps = 1e-6
    df_plot = agg.copy()
    df_plot["mean_elapsed"] = df_plot["mean_elapsed"].clip(lower=eps)

    plt.figure(figsize=(10, 6))
    plotted = False
    for algo, g in df_plot.groupby("algo_display"):
        if g.empty:
            continue
        plt.errorbar(g["n"].values, g["mean_elapsed"].values,
                     yerr=g["std_elapsed"].fillna(0).values, marker="o", label=algo)
        plotted = True
    if plotted:
        if df_plot["n"].max() / max(1, df_plot["n"].min()) > 20:
            plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Размер матрицы, n")
        plt.ylabel("Время, сек (логарифмическая шкала)")
        plt.title("Затраченное время в зависимости от размера матрицы"
                  + (f" — {title_suffix}" if title_suffix else ""))
        plt.grid(True, which="both", ls="--", alpha=0.4)
        _safe_legend()
    out_png2 = os.path.join(outdir, f"elapsed_vs_n{('_' + title_suffix) if title_suffix else ''}.png")
    plt.tight_layout();
    plt.savefig(out_png2);
    plt.close()
    print("Сохранён график:", out_png2)

    # --- 3) Boxplot ошибки по каждому n ---
    for n in sorted(df["n"].unique()):
        sub = df[df["n"] == n]
        if sub.empty:
            continue
        labels, data = [], []
        for algo in sorted(sub["algo_display"].unique()):
            arr = sub[sub["algo_display"] == algo]["rel_error"].dropna().values
            if arr.size == 0:
                continue
            labels.append(algo)
            data.append(100.0 * arr)
        if not data:
            continue
        plt.figure(figsize=(9, 5))
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.ylabel("Относительная ошибка, %")
        plt.title(f"Распределение относительной ошибки по алгоритмам (n={n})"
                  + (f" — {title_suffix}" if title_suffix else ""))
        plt.grid(True, axis="y", ls="--", alpha=0.3)
        outpng = os.path.join(outdir, f"box_rel_error_n{n}{('_' + title_suffix) if title_suffix else ''}.png")
        plt.tight_layout();
        plt.savefig(outpng);
        plt.close()
    print("Сохранены boxplot-графики в", outdir)

    # --- 4) «Скорость vs Точность» (разброс) ---
    plt.figure(figsize=(10, 6))
    plotted = False
    for algo in sorted(df["algo_display"].unique()):
        sub = df[df["algo_display"] == algo].copy()
        if sub.empty:
            continue
        sub["elapsed_plot"] = sub["elapsed"].clip(lower=eps)
        plt.scatter(sub["elapsed_plot"], 100.0 * sub["rel_error"], label=algo, alpha=0.85, s=35)
        plotted = True
    if plotted:
        plt.xscale("log")
        plt.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.7)
        plt.xlabel("Время, сек (логарифмическая шкала)")
        plt.ylabel("Относительная ошибка, %")
        plt.title("Компромисс «скорость–точность» (меньше — лучше)"
                  + (f" — {title_suffix}" if title_suffix else ""))
        plt.grid(True, which="both", ls="--", alpha=0.4)
        _safe_legend()
    out_png3 = os.path.join(outdir, f"speed_vs_accuracy{('_' + title_suffix) if title_suffix else ''}.png")
    plt.tight_layout();
    plt.savefig(out_png3);
    plt.close()
    print("Сохранён график:", out_png3)


# ---------------- основной сценарий ----------------

def _prepare_outdir(outdir: str) -> None:
    Path(outdir).mkdir(parents=True, exist_ok=True)


def _run_mode(mode_name: str, sizes: List[int], repeats: int, timelimit: float, outroot: str,
              mode_gen: str, verbose: bool) -> None:
    outdir = os.path.join(outroot, mode_name)
    _prepare_outdir(outdir)
    print(f"\n=== Запуск режима {mode_name} ===")
    print(f"Размеры: {sizes}, повторов на размер: {repeats}, лимит времени: {timelimit} с, выход: {outdir}")

    # 1) Истинные оптимумы
    print("\n[Шаг 1/2] Считаю истинные оптимумы (библиотечный Венгерский, без лимита)…")
    opt_map: Dict[Tuple[int, int], float] = {}
    total = len(sizes) * repeats
    cur = 0
    for n in sizes:
        for seed in range(repeats):
            cur += 1
            print(f"  [{cur}/{total}] n={n} seed={seed} -> ", end="", flush=True)
            opt = _run_unlimited_optimal_cost(n=n, seed=seed, mode=mode_gen)
            if opt is None or not np.isfinite(opt):
                print("ошибка")
            else:
                print(f"ok (opt={opt:.6f})")
                opt_map[(n, seed)] = opt
    pd.DataFrame(
        [{"n": n, "seed": s, "opt_cost_true": c} for (n, s), c in sorted(opt_map.items())]
    ).to_csv(os.path.join(outdir, "opt_costs_true.csv"), index=False)

    # 2) Ограниченные запуски бенчмарка
    print("\n[Шаг 2/2] Ограниченные по времени запуски (все алгоритмы) c параметрами ACO по размеру…")
    cur = 0
    for n in sizes:
        aco_params = get_aco_params_for_n(n)
        for seed in range(repeats):
            cur += 1
            print(f"  [{cur}/{total}] n={n} seed={seed}")
            _ = _run_benchmark_unified(n=n, seed=seed, timelimit=timelimit,
                                       outdir=outdir, mode=mode_gen,
                                       aco_params=aco_params, verbose=verbose)
            time.sleep(0.05)

    # 3) Пересчёт ошибок и визуализации
    summary_csv = os.path.join(outdir, "summary.csv")
    if not os.path.exists(summary_csv):
        print(f"[WARN] Не найден {summary_csv}")
        return
    df = _recompute_errors_with_true_opt(summary_csv, opt_map)
    _save_agg_and_plots(df, outdir, title_suffix=mode_name)
    print(f"Готово для режима {mode_name}. Результаты: {outdir}")


def parse_sizes_arg(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Сравнение ACO vs Венгерский (2 режима); ACO с пер-размерными параметрами")
    ap.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)),
                    help="Список n через запятую")
    ap.add_argument("--repeats", type=int, default=5, help="Число повторов на каждый n (seeds = 0..repeats-1)")
    ap.add_argument("--outroot", type=str, default="out", help="Папка для результатов")
    ap.add_argument("--mode_gen", type=str, default="uniform", help="Генератор матриц")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--time_A", type=float, default=2.0, help="Лимит времени для режима A, сек")
    ap.add_argument("--time_B", type=float, default=0.1, help="Лимит времени для режима B, сек")
    args = ap.parse_args()

    sizes = parse_sizes_arg(args.sizes)
    print("Размеры:", sizes)
    print("Повторов:", args.repeats)
    print("Выходная папка:", args.outroot)
    os.makedirs(args.outroot, exist_ok=True)

    _run_mode("modeA", sizes, args.repeats, args.time_A, args.outroot, mode_gen=args.mode_gen, verbose=args.verbose)
    _run_mode("modeB", sizes, args.repeats, args.time_B, args.outroot, mode_gen=args.mode_gen, verbose=args.verbose)

    # Общий отчёт по двум режимам
    combined = []
    for mode_name in ("modeA", "modeB"):
        pth = Path(args.outroot) / mode_name / "summary.csv"
        if pth.exists():
            df = pd.read_csv(str(pth))
            df["exp_mode"] = mode_name
            combined.append(df)
    if combined:
        comb = pd.concat(combined, ignore_index=True)
        comb_csv = os.path.join(args.outroot, "summary_combined_modes.csv")
        comb.to_csv(comb_csv, index=False)
        print("Сохранён общий CSV:", comb_csv)

        # Сравнение режимов по средней ошибке
        agg = comb.groupby(["exp_mode", "algo_display", "n"]).rel_error.mean().reset_index()
        plt.figure(figsize=(10, 6))
        plotted = False
        for algo in sorted(agg["algo_display"].unique()):
            for mode_name in ("modeA", "modeB"):
                sub = agg[(agg["algo_display"] == algo) & (agg["exp_mode"] == mode_name)]
                if sub.empty:
                    continue
                plt.plot(sub["n"], 100.0 * sub["rel_error"], marker="o",
                         linestyle="--" if mode_name == "modeB" else "-",
                         label=f"{algo} — {mode_name}")
                plotted = True
        if plotted:
            if comb["n"].max() / max(1, comb["n"].min()) > 20:
                plt.xscale("log")
            plt.axhline(0.0, color="k", linestyle="--", linewidth=1, alpha=0.7)
            plt.xlabel("Размер матрицы, n")
            plt.ylabel("Средняя относительная ошибка, %  (отн. библиотечного Венгерского)")
            plt.title("Сравнение режимов: средняя ошибка по алгоритмам")
            plt.grid(True, which="both", ls="--", alpha=0.4)
            _safe_legend()
        out_png = os.path.join(args.outroot, "modeA_vs_modeB_rel_error.png")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print("Сохранён график:", out_png)

    print("Готово.")


if __name__ == "__main__":
    main()
