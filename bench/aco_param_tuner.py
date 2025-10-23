# bench/aco_param_tuner.py
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# --- ACO import (без правок ваших файлов) ------------------------------------
# Пытаемся импортировать ваш класс/параметры ACO из проекта, не меняя их.
# Поддерживаем разные возможные расположения.
ACO_IMPORT_OK = False
try:
    from aco import AntColonyAssignment, ACOParams  # type: ignore
    ACO_IMPORT_OK = True
except Exception:
    try:
        from ant_colony.aco import AntColonyAssignment, ACOParams  # type: ignore
        ACO_IMPORT_OK = True
    except Exception:
        try:
            from ant_colony.core import AntColonyAssignment, ACOParams  # type: ignore
            ACO_IMPORT_OK = True
        except Exception as e:
            ACO_IMPORT_OK = False
            ACO_IMPORT_ERR = e

# SciPy (для эталонного решения Венгерским методом)
try:
    from scipy.optimize import linear_sum_assignment
except Exception as e:
    print("[ERROR] SciPy не найден. Установите scipy (`pip install scipy`).", file=sys.stderr)
    raise

# --- Константы ----------------------------------------------------------------
DEFAULT_SIZES = [25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 2000, 3000, 4000, 5000]

# Границы пространства поиска гиперпараметров (обоснованы эмпирикой и практикой ACO)
ALPHA_RANGE = (0.7, 1.6)
BETA_RANGE  = (2.0, 6.0)
RHO_RANGE   = (0.05, 0.4)
Q0_RANGE    = (0.05, 0.9)

# --- Вспомогательные структуры ------------------------------------------------
@dataclass
class TrialResult:
    n: int
    seed: int
    ants: int
    alpha: float
    beta: float
    rho: float
    q0: float
    local_search: str
    ls_iters: int
    max_iters: int
    tlim: float
    aco_cost: float
    opt_cost: float
    rel_error: float
    elapsed_trial: float

# --- Утилиты ------------------------------------------------------------------
def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def weighted_equal_split(total_seconds: float, count: int) -> List[float]:
    """Равное деление общего бюджета на count кусков."""
    if count <= 0:
        return []
    base = total_seconds / count
    return [base for _ in range(count)]

def choose_ants_range(n: int) -> Tuple[int, int]:
    """Эмпирически разумные диапазоны ants в зависимости от n."""
    if n <= 75:
        return (10, 40)
    if n <= 150:
        return (12, 60)
    if n <= 400:
        return (14, 80)
    if n <= 1200:
        return (16, 100)
    return (18, 120)

def choose_ls_iters_range(n: int) -> Tuple[int, int]:
    """Диапазоны количества итераций локального поиска."""
    if n <= 100:
        return (80, 300)
    if n <= 400:
        return (50, 200)
    if n <= 1200:
        return (30, 150)
    return (20, 120)

def suggest_local_search(n: int, per_trial_tlim: float) -> str:
    """
    Когда стоит включать локальный поиск 2swap:
    - для небольших n — чаще,
    - для очень маленького tlim — иногда лучше выключить.
    """
    if per_trial_tlim < 0.08:
        return ""  # слишком мало времени на пользу от LS
    if n <= 100:
        return "2swap"
    if n <= 400:
        return "2swap"
    # для больших матриц включаем реже; но пробуем в рандомизированном поиске
    return "2swap" if np.random.rand() < 0.5 else ""

def choose_trial_target(n: int) -> int:
    """Желаемое число проб на размер (будет скорректировано реальным бюджетом)."""
    if n <= 75:
        return 160
    if n <= 150:
        return 120
    if n <= 400:
        return 80
    if n <= 1200:
        return 48
    return 28

def sample_hyperparams(n: int, per_trial_tlim: float, rng: np.random.Generator) -> Dict:
    ants_lo, ants_hi = choose_ants_range(n)
    ls_lo, ls_hi     = choose_ls_iters_range(n)

    # непрерывные параметры, округляем до 1 знака для стабильности
    alpha = float(np.round(rng.uniform(*ALPHA_RANGE), 1))
    beta  = float(np.round(rng.uniform(*BETA_RANGE), 1))
    rho   = float(np.round(rng.uniform(*RHO_RANGE), 2))
    q0    = float(np.round(rng.uniform(*Q0_RANGE), 2))

    ants  = int(rng.integers(ants_lo, ants_hi + 1))

    # режим локального поиска
    local_search = suggest_local_search(n, per_trial_tlim)
    ls_iters     = int(rng.integers(ls_lo, ls_hi + 1)) if local_search == "2swap" else 0

    # max_iters — верхняя «страховка»; реально ограничивает time_limit
    # масштабируем очень грубо от n, чтобы не портить поведение вашего ACO
    if n <= 100:
        max_iters = 200
    elif n <= 400:
        max_iters = 150
    elif n <= 1200:
        max_iters = 120
    else:
        max_iters = 100

    return dict(
        ants=ants, alpha=alpha, beta=beta, rho=rho, q0=q0,
        local_search=local_search, ls_iters=ls_iters,
        max_iters=max_iters
    )

def compute_opt_cost(C: np.ndarray) -> float:
    r, c = linear_sum_assignment(C)
    return float(C[r, c].sum())

def run_aco_once(C: np.ndarray, hp: Dict, tlim: float, seed: int) -> float:
    """Запуск ACO на одной матрице C с заданными гиперпараметрами и лимитом времени (сек). Возвращает стоимость решения."""
    if not ACO_IMPORT_OK:
        raise RuntimeError(f"Не удалось импортировать ACO из проекта: {ACO_IMPORT_ERR}")

    params = ACOParams(
        num_ants=hp["ants"],
        alpha=hp["alpha"],
        beta=hp["beta"],
        rho=hp["rho"],
        q0=hp["q0"],
        tau0=None,
        Q=1.0,
        local_search=hp["local_search"],
        ls_iters=hp["ls_iters"],
        max_iters=hp["max_iters"],
        time_limit=float(tlim),
        seed=int(seed),
        verbose=False,
    )
    aco = AntColonyAssignment(params=params)
    p, cost, _info = aco.solve(C)
    return float(cost)

def write_progress_row(path: str, row: Dict, header: Optional[List[str]]) -> None:
    """Дописывает строку в CSV (создаёт файл с заголовком, если его нет)."""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

def plot_trends(best_rows: List[Dict], out_png: str) -> None:
    """Рисуем простой график зависимостей гиперпараметров от n."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib недоступен, пропущу построение графика: {e}")
        return

    if not best_rows:
        print("[WARN] нет данных для графика трендов.")
        return

    best_rows_sorted = sorted(best_rows, key=lambda r: r["n"])
    ns = [r["n"] for r in best_rows_sorted]

    def take(key: str):
        return [r[key] for r in best_rows_sorted]

    fig = plt.figure(figsize=(11, 9))
    fig.suptitle("ACO tuned hyperparameters vs matrix size (n)")

    # ants
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.plot(ns, take("ants"), marker="o")
    ax1.set_title("num_ants")
    ax1.set_xlabel("n")
    ax1.set_ylabel("ants")

    # alpha
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(ns, take("alpha"), marker="o")
    ax2.set_title("alpha")
    ax2.set_xlabel("n")

    # beta
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(ns, take("beta"), marker="o")
    ax3.set_title("beta")
    ax3.set_xlabel("n")

    # rho
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(ns, take("rho"), marker="o")
    ax4.set_title("rho")
    ax4.set_xlabel("n")

    # q0
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(ns, take("q0"), marker="o")
    ax5.set_title("q0")
    ax5.set_xlabel("n")

    # ls_iters (0 если LS выключен)
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(ns, take("ls_iters"), marker="o")
    ax6.set_title("ls_iters")
    ax6.set_xlabel("n")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# --- Основная логика ----------------------------------------------------------
def main():
    if not ACO_IMPORT_OK:
        print("[ERROR] Не удалось импортировать ваш ACO. Убедитесь, что модуль доступен в PYTHONPATH.")
        print("Поддерживаемые варианты импорта: `from aco import ...`, `from ant_colony.aco import ...`, `from ant_colony.core import ...`")
        print(f"Исключение: {ACO_IMPORT_ERR}")
        sys.exit(1)

    ap = argparse.ArgumentParser(description="Hyperparameter tuning for ACO (assignment problem).")
    ap.add_argument("--total_hours", type=float, default=12.0, help="Общее время работы тюнера (часы). Например: 12")
    ap.add_argument("--out_dir", type=str, default="out", help="Каталог для результатов.")
    ap.add_argument("--sizes", type=int, nargs="*", default=DEFAULT_SIZES, help="Список размеров матриц n.")
    ap.add_argument("--seed_start", type=int, default=0, help="Базовый seed для генерации матриц.")
    ap.add_argument("--min_per_size", type=float, default=5.0, help="Минимум секунд на каждый размер (страховка).")
    ap.add_argument("--matrix_mode", type=str, default="uniform", choices=["uniform"], help="Тип генерации матрицы.")
    args = ap.parse_args()

    ensure_dirs(args.out_dir)
    total_seconds = max(0.0, float(args.total_hours) * 3600.0)
    sizes: List[int] = list(args.sizes)
    n_sizes = len(sizes)

    print("=== ACO hyperparameter tuning ===")
    print(f"Total budget ~ {args.total_hours:.2f} hours; {int(total_seconds)} seconds")
    print(f"Sizes: {sizes}")

    if n_sizes == 0 or total_seconds <= 0.0:
        print("[ERROR] Пустой список размеров или нулевой бюджет времени.")
        sys.exit(2)

    # Равное деление бюджета по размерам
    split = weighted_equal_split(total_seconds, n_sizes)
    # Учитываем минимальный бюджет на размер (если общий бюджет слишком маленький — предупредим)
    need = sum(max(s, args.min_per_size) for s in split)
    if need > total_seconds:
        print(f"[WARN] Общий бюджет ({total_seconds:.1f}s) меньше суммы минимальных ({need:.1f}s).")
        print("      Будет использоваться равный сплит и строгий общий дедлайн; некоторые размеры могут не успеть.")
    print("Per-size budgets (seconds, equal split):")
    for n, sec in zip(sizes, split):
        print(f"  n={n}: {sec:.1f}s")

    # Пути сохранения
    progress_csv = os.path.join(args.out_dir, "aco_tuning_progress.csv")
    best_csv     = os.path.join(args.out_dir, "aco_tuned_params.csv")
    trends_png   = os.path.join(args.out_dir, "aco_param_trends.png")

    # Заголовки CSV
    progress_header = [
        "time", "n", "seed",
        "ants", "alpha", "beta", "rho", "q0", "local_search", "ls_iters", "max_iters", "tlim",
        "aco_cost", "opt_cost", "rel_error", "elapsed_trial"
    ]
    best_header = [
        "n", "seed", "ants", "alpha", "beta", "rho", "q0", "local_search",
        "ls_iters", "max_iters", "tlim", "aco_cost", "opt_cost", "rel_error", "trials"
    ]

    # Глобальный дедлайн
    global_deadline = time.time() + total_seconds

    best_rows: List[Dict] = []
    global_start = time.time()

    for idx, (n, per_n_seconds) in enumerate(zip(sizes, split), start=1):
        if time.time() >= global_deadline:
            print("\nГлобальный дедлайн достигнут до обработки всех размеров. Завершаю.")
            break

        # Пер-размерный дедлайн: либо равный сплит, либо остаток до глобального
        per_deadline = min(global_deadline, time.time() + max(per_n_seconds, 0.0))
        per_start = time.time()

        print(f"\n[{idx}/{n_sizes}] Tune n={n} with budget ~{max(0.0, per_deadline - time.time()):.1f}s")

        # Генерация одной матрицы на размер (фиксируем seed для честного сравнения конфигураций)
        mat_seed = args.seed_start + idx * 1000
        rng = np.random.default_rng(mat_seed)
        if args.matrix_mode == "uniform":
            C = rng.random((n, n), dtype=np.float64)
        else:
            # На будущее — можно добавить другие режимы генерации
            C = rng.random((n, n), dtype=np.float64)

        # Эталон (венгерский) — считаем один раз для данного n
        try:
            opt_cost = compute_opt_cost(C)
        except Exception as e:
            print(f"[ERROR] SciPy LSA failed on n={n}: {e}")
            break

        # Выбор желаемого числа проб и оценка tlim на пробу (адаптивно от бюджета)
        target_trials = choose_trial_target(n)
        # 70% от пер-размерного времени пойдёт на реальные прогоны (остальное — накладные расходы)
        if per_deadline > time.time():
            raw_slot = 0.70 * (per_deadline - time.time()) / max(1, target_trials)
        else:
            raw_slot = 0.0
        # ограничиваем tlim, чтобы он был разумный
        per_trial_tlim = float(np.clip(raw_slot, 0.05, 2.0))

        best_trial: Optional[TrialResult] = None
        trials_done = 0
        ema_trial_sec = None  # экспоненциальная скользящая для учёта реальной длительности пробы

        # Основной цикл проб
        while True:
            now = time.time()
            if now >= per_deadline:
                break
            if now >= global_deadline:
                print("  -> глобальный дедлайн в процессе тюнинга размера. Останавливаюсь.")
                break

            # Прогноз времени на ещё одну пробу (используем EMA фактической длительности)
            if ema_trial_sec is None:
                expect_next = per_trial_tlim * 1.35  # грубая оценка
            else:
                expect_next = min(max(ema_trial_sec * 1.10, per_trial_tlim * 1.15), 3.5 * per_trial_tlim)

            if now + expect_next > per_deadline:
                # не успеваем ещё одну пробу — выходим
                break

            # семя для ACO (перепробуем разные)
            aco_seed = mat_seed + trials_done

            # семплируем гиперпараметры
            hp = sample_hyperparams(n, per_trial_tlim, rng)

            # Запуск одной ACO-пробы
            t0 = time.time()
            try:
                aco_cost = run_aco_once(C, hp, per_trial_tlim, aco_seed)
            except Exception as e:
                # если что-то упало — логируем и идём дальше
                aco_cost = float("inf")
            t1 = time.time()

            elapsed_trial = t1 - t0
            ema_trial_sec = elapsed_trial if ema_trial_sec is None else 0.7 * ema_trial_sec + 0.3 * elapsed_trial

            rel_error = (aco_cost / opt_cost) - 1.0 if np.isfinite(aco_cost) else float("inf")

            # Лог + запись в прогресс
            pr = TrialResult(
                n=n, seed=aco_seed,
                ants=hp["ants"], alpha=hp["alpha"], beta=hp["beta"], rho=hp["rho"], q0=hp["q0"],
                local_search=hp["local_search"], ls_iters=hp["ls_iters"], max_iters=hp["max_iters"], tlim=per_trial_tlim,
                aco_cost=aco_cost, opt_cost=opt_cost, rel_error=rel_error, elapsed_trial=elapsed_trial
            )
            write_progress_row(progress_csv, {
                "time": now_str(), **asdict(pr)
            }, header=progress_header)

            # Обновление лучшего
            if (best_trial is None) or (rel_error < best_trial.rel_error):
                best_trial = pr

            trials_done += 1

            # Небольшой прогресс-лог раз в ~10 проб (или на первых)
            if (trials_done <= 5) or (trials_done % 10 == 0):
                best_rel = best_trial.rel_error if best_trial else float("inf")
                print(f"    trial={trials_done:<4d}  last_rel={rel_error:>7.4f}  best_rel={best_rel:>7.4f}  "
                      f"tlim={per_trial_tlim:.2f}s  used={time.time()-per_start:.1f}s / {max(0.0, per_deadline-per_start):.1f}s")

        # Итог по размеру
        if best_trial is None:
            print(f"  -> Не успели выполнить ни одной пробы для n={n}.")
        else:
            print(f"  -> best rel_error={best_trial.rel_error:.4f} | "
                  f"params: ants={best_trial.ants}, alpha={best_trial.alpha}, beta={best_trial.beta}, "
                  f"rho={best_trial.rho}, q0={best_trial.q0}, ls_iters={best_trial.ls_iters}, "
                  f"local_search='{best_trial.local_search}', max_iters={best_trial.max_iters}, "
                  f"tlim={best_trial.tlim:.2f}s | trials={trials_done}")

            # Сохраняем лучшую строку в общий CSV
            best_row = dict(
                n=best_trial.n, seed=best_trial.seed,
                ants=best_trial.ants, alpha=best_trial.alpha, beta=best_trial.beta,
                rho=best_trial.rho, q0=best_trial.q0, local_search=best_trial.local_search,
                ls_iters=best_trial.ls_iters, max_iters=best_trial.max_iters,
                tlim=best_trial.tlim, aco_cost=best_trial.aco_cost, opt_cost=best_trial.opt_cost,
                rel_error=best_trial.rel_error, trials=trials_done
            )
            write_progress_row(best_csv, best_row, header=best_header)
            best_rows.append(best_row)

    # Построение графика трендов
    plot_trends(best_rows, trends_png)

    total_used = time.time() - global_start
    print("\nSaved results to:")
    print(f"  {best_csv}")
    print(f"  {trends_png}")
    print(f"  progress: {progress_csv}")
    print(f"Total used time: {total_used:.1f}s")

if __name__ == "__main__":
    main()
