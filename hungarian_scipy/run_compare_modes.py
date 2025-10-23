#!/usr/bin/env python3
"""
run_compare_modes.py

Запускает серию экспериментов сравнительного анализа:
 - режим A (достаточно времени)
 - режим B (жёсткий таймаут "никто не успевает")

Использует bench/benchmark_unified.py (скрипт проекта).
Результаты: out/modeA/* и out/modeB/* (csv + png).
"""
import argparse
import subprocess
import sys
import os
import shutil
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PY = sys.executable

DEFAULT_SIZES = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000, 3000, 4000, 5000]


def run_one(n, seed, timelimit, outdir, mode="uniform", ants=80, iters=2000, local_search="2swap", verbose=False):
    """Выполнить единичный запуск bench.benchmark_unified --algo all."""
    cmd = [
        PY, "-m", "bench.benchmark_unified",
        "--algo", "all",
        "--n", str(n),
        "--mode", mode,
        "--seed", str(seed),
        "--time_limit", str(timelimit),
        "--ants", str(ants),
        "--iters", str(iters),
        "--local_search", local_search,
        "--outdir", outdir,
        "--out_csv", os.path.join(outdir, "summary.csv"),
        # не сохраняем history_csv по умолчанию (чтобы не плодить файлы); можно задать отдельно
    ]
    if verbose:
        print("RUN:", " ".join(cmd))
    # Запуск синхронно; stdout/err будут печататься в консоль
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # печатаем лог из процесса (полезно для отладки)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"[WARN] process returned code {proc.returncode} for n={n}, seed={seed}")
    return proc.returncode


def prepare_outdir(outdir):
    p = Path(outdir)
    if p.exists():
        # Удаляем старые summary.csv чтобы не мешать
        s = p / "summary.csv"
        if s.exists():
            s.unlink()
    else:
        p.mkdir(parents=True, exist_ok=True)


def aggregate_and_plot(summary_csv, outdir, title_suffix=""):
    df = pd.read_csv(summary_csv)
    # приведение типов
    df['n'] = pd.to_numeric(df['n'], errors='coerce')
    df['rel_error'] = pd.to_numeric(df['rel_error'], errors='coerce')
    df['elapsed'] = pd.to_numeric(df['elapsed'], errors='coerce')

    # Агрегация: mean/std по (algo, n)
    agg = df.groupby(['algo', 'n']).agg(
        mean_rel_error=('rel_error', 'mean'),
        std_rel_error=('rel_error', 'std'),
        mean_elapsed=('elapsed', 'mean'),
        std_elapsed=('elapsed', 'std'),
        count=('rel_error', 'count')
    ).reset_index()

    agg_csv = os.path.join(outdir, f"agg_stats{('_' + title_suffix) if title_suffix else ''}.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregated stats to {agg_csv}")

    # График: mean rel_error vs n
    plt.figure(figsize=(9, 6))
    for algo, g in agg.groupby('algo'):
        plt.errorbar(g['n'], g['mean_rel_error'], yerr=g['std_rel_error'].fillna(0), label=algo, marker='o')
    plt.xscale('log' if agg['n'].max() / agg['n'].min() > 20 else 'linear')
    plt.xlabel('n')
    plt.ylabel('mean relative error')
    plt.title('Relative error vs n' + (f" — {title_suffix}" if title_suffix else ""))
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    out_png = os.path.join(outdir, f"rel_error_vs_n{('_' + title_suffix) if title_suffix else ''}.png")
    plt.tight_layout();
    plt.savefig(out_png);
    plt.close()
    print("Saved", out_png)

    # График: mean elapsed vs n (лог-шкала Y)
    plt.figure(figsize=(9, 6))
    for algo, g in agg.groupby('algo'):
        plt.errorbar(g['n'], g['mean_elapsed'], yerr=g['std_elapsed'].fillna(0), label=algo, marker='o')
    plt.xscale('log' if agg['n'].max() / agg['n'].min() > 20 else 'linear')
    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('mean elapsed (s) [log scale]')
    plt.title('Elapsed time vs n' + (f" — {title_suffix}" if title_suffix else ""))
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.4)
    out_png2 = os.path.join(outdir, f"elapsed_vs_n{('_' + title_suffix) if title_suffix else ''}.png")
    plt.tight_layout();
    plt.savefig(out_png2);
    plt.close()
    print("Saved", out_png2)

    # Boxplots rel_error per n
    for n in sorted(df['n'].unique()):
        sub = df[df['n'] == n]
        if sub.empty: continue
        labels = []
        data = []
        for algo in sorted(sub['algo'].unique()):
            arr = sub[sub['algo'] == algo]['rel_error'].dropna()
            if len(arr) == 0: continue
            labels.append(algo)
            data.append(arr.values)
        if not data: continue
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels, showmeans=True)
        plt.ylabel('rel_error')
        plt.title(f'rel_error distribution for n={n} {title_suffix}')
        plt.tight_layout()
        outpng = os.path.join(outdir, f"box_rel_error_n{n}{('_' + title_suffix) if title_suffix else ''}.png")
        plt.savefig(outpng);
        plt.close()
    print("Saved boxplots for individual n in", outdir)

    # Доп: сохранить исходный summary в формате parquet для быстрой аналитики
    try:
        pq = os.path.join(outdir, f"summary_parquet{('_' + title_suffix) if title_suffix else ''}.parquet")
        df.to_parquet(pq, index=False)
        print("Saved parquet:", pq)
    except Exception:
        pass

    return agg


def run_mode(mode_name, sizes, repeats, timelimit, outroot, mode_gen="uniform", ants=80, iters=2000,
             local_search="2swap", verbose=False):
    outdir = os.path.join(outroot, mode_name)
    prepare_outdir(outdir)
    print(f"Running mode {mode_name}: sizes={sizes}, repeats={repeats}, time_limit={timelimit}, outdir={outdir}")
    total = len(sizes) * repeats
    cur = 0
    for n in sizes:
        for seed in range(repeats):
            cur += 1
            print(f"[{mode_name}] {cur}/{total} : n={n}, seed={seed}")
            # call unified benchmark; it appends to outdir/summary.csv
            rc = run_one(n=n, seed=seed, timelimit=timelimit, outdir=outdir, mode=mode_gen,
                         ants=ants, iters=iters, local_search=local_search, verbose=verbose)
            # small sleep to avoid tight loop on some systems
            time.sleep(0.1)
    # анализ и графики
    summary_csv = os.path.join(outdir, "summary.csv")
    if not os.path.exists(summary_csv):
        print("No summary.csv produced for mode", mode_name)
        return
    agg = aggregate_and_plot(summary_csv, outdir, title_suffix=mode_name)
    print(f"Completed mode {mode_name}. Results in {outdir}")


def parse_sizes_arg(sizes_arg):
    if sizes_arg:
        parts = [int(x) for x in sizes_arg.split(",") if x.strip()]
        return parts
    return DEFAULT_SIZES


def main():
    p = argparse.ArgumentParser(description="Run comparative experiments: Hungarian vs ACO, Mode A/B")
    p.add_argument("--sizes", type=str, default=",".join(map(str, DEFAULT_SIZES)),
                   help="Comma-separated sizes list. Default: adaptive list from 25..5000.")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--outroot", type=str, default="out")
    p.add_argument("--mode_gen", type=str, default="uniform", help="Matrix generator mode")
    p.add_argument("--ants", type=int, default=80)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--local_search", type=str, default="2swap")
    p.add_argument("--verbose", action="store_true")
    # Time limits for Mode A and Mode B
    p.add_argument("--time_A", type=float, default=2.0, help="Mode A (enough time) default 2.0s")
    p.add_argument("--time_B", type=float, default=0.1, help="Mode B (tight) default 0.1s")
    args = p.parse_args()

    sizes = parse_sizes_arg(args.sizes)
    print("Sizes:", sizes)
    print("Repeats:", args.repeats)
    print("Out root:", args.outroot)
    os.makedirs(args.outroot, exist_ok=True)

    # Run Mode A (sufficient time)
    run_mode("modeA", sizes, args.repeats, args.time_A, args.outroot, mode_gen=args.mode_gen,
             ants=args.ants, iters=args.iters, local_search=args.local_search, verbose=args.verbose)

    # Run Mode B (tight time)
    run_mode("modeB", sizes, args.repeats, args.time_B, args.outroot, mode_gen=args.mode_gen,
             ants=args.ants, iters=args.iters, local_search=args.local_search, verbose=args.verbose)

    # Combine summary CSVs for easier comparison
    combined = []
    for mode_name in ("modeA", "modeB"):
        pth = Path(args.outroot) / mode_name / "summary.csv"
        if pth.exists():
            df = pd.read_csv(str(pth))
            df['exp_mode'] = mode_name
            combined.append(df)
    if combined:
        comb = pd.concat(combined, ignore_index=True)
        combined_csv = os.path.join(args.outroot, "summary_combined_modes.csv")
        comb.to_csv(combined_csv, index=False)
        print("Saved combined CSV:", combined_csv)

        # quick plot comparing modeA vs modeB for rel_error (averaged)
        agg = comb.groupby(['exp_mode', 'algo', 'n']).rel_error.mean().reset_index()
        plt.figure(figsize=(9, 6))
        for algo in sorted(agg['algo'].unique()):
            for mode_name in ['modeA', 'modeB']:
                sub = agg[(agg['algo'] == algo) & (agg['exp_mode'] == mode_name)]
                if sub.empty: continue
                plt.plot(sub['n'], sub['rel_error'], marker='o', linestyle='--' if mode_name == 'modeB' else '-',
                         label=f"{algo}-{mode_name}")
        plt.xscale('log' if comb['n'].max() / comb['n'].min() > 20 else 'linear')
        plt.xlabel('n');
        plt.ylabel('mean rel_error');
        plt.title('ModeA vs ModeB rel_error by algo')
        plt.legend();
        plt.grid(True)
        out_png = os.path.join(args.outroot, "modeA_vs_modeB_rel_error.png")
        plt.tight_layout();
        plt.savefig(out_png);
        plt.close()
        print("Saved", out_png)

    print("All done.")


if __name__ == "__main__":
    main()
