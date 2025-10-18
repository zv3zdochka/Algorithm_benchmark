# Муравьиный алгоритм (ACO) для задачи о назначениях

Этот репозиторий — реализация **Ant Colony Optimization (ACO)**
для линейной задачи о назначениях (минимизация суммарной стоимости). Реализация поддерживает режим *anytime*
(по времени или по числу итераций), опциональный локальный 2‑swap, CLI, юнит‑тесты,
бенчмарк с историей и графиками.

## Что внутри
```
ant_colony/
  __init__.py          — экспорт AntColonyAssignment, ACOParams
  aco.py               — ядро ACO
  utils.py             — генератор матриц, брутфорс для tiny, greedy‑базлайн, сохранение history.csv
ant_colony/run.py      — CLI‑раннер
bench/benchmark.py     — бенчмарк (history.csv + convergence.png)
tests/test_small.py    — юнит‑тесты
data/                  — примеры матриц (CSV)
README.md              — краткая документация
requirements.txt       — зависимости
```

## Установка
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Быстрый старт
Запуск на случайной матрице 60×60 с бюджетом 1.0 секунды и фиксированным seed:
```bash
python -m ant_colony.run --n 60 --mode uniform --time_limit 1.0 --seed 42
```
Запуск на CSV:
```bash
python -m ant_colony.run --csv data/tiny_4x4_diagonal.csv --iters 300 --ants 30 --seed 7
```

### Формат ввода/вывода
- **Вход:** квадратная матрица стоимостей `C` (CSV или NPY, либо `numpy.ndarray` в API).
- **Выход:** перестановка `perm` длины `n` (строка *i* назначается на столбец `perm[i]`),
  суммарная стоимость `best_cost`, и `info.history` с метриками по итерациям.

## Параметры (основные)
- `num_ants` — сколько муравьёв на итерацию (обычно ≈ *n*).
- `alpha`, `beta` — вес феромона и эвристики (типично `alpha=1`, `beta=3`).
- `rho` — испарение (например `0.1`).
- `q0` — доля жадных шагов (например `0.2`).
- `time_limit` или `max_iters` — бюджет.
- `local_search="2swap"` — быстрый локальный подъём.
- `seed` — воспроизводимость.

## Примеры использования API
```python
import numpy as np
from ant_colony import AntColonyAssignment, ACOParams

C = np.random.rand(50, 50)
params = ACOParams(num_ants=60, max_iters=400, seed=42, local_search="2swap")
aco = AntColonyAssignment(params)
perm, best_cost, info = aco.solve(C)

# perm[i] — назначенный столбец для строки i (вектор "цель → траектория")
```

## 4) Автотесты и прогоны на больших размерах

### Юнит‑тесты
```bash

python -m unittest tests/test_small.py -v
```

### Большие случайные матрицы

**Через CLI‑раннер (`ant_colony/run.py`):**
```bash
# 100×100, бюджет 1.0с, история в CSV
python -m ant_colony.run --n 100 --mode uniform --time_limit 1.0 --seed 42 --history_csv out/history_100.csv
```

**Через бенч‑скрипт (`bench/benchmark.py`):**
```bash
# соберёт history.csv и convergence.png
python -m bench.benchmark --n 300 --time_limit 1.0 --iters 800 --ants 300 --outdir bench_out
```

**Пояснения:**
- `--mode` в генераторе (`utils.generate_matrix`): `uniform | normal | diag_dom | block`.
- Для больших n удобно задавать `--time_limit`, чтобы сравнения были в одинаковом окне времени.

## 5) Фиксация результатов: таблицы и графики

**История сходимости (CSV):**
- Любой запуск с `--history_csv` пишет таблицу:
  `iter, elapsed, iter_best_cost, best_cost, mean_cost, std_cost`

Пример:
```bash
python -m ant_colony.run --n 200 --time_limit 1.5 --history_csv out/history_200.csv
```

**График сходимости:**
```bash
python -m bench.benchmark --n 120 --time_limit 1.5 --outdir bench_out
# → bench_out/history.csv, bench_out/convergence.png
```