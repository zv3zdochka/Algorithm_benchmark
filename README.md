# Модуль для тестирования алгоритмов назначения по скорости и качеству

Этот проект — удобный стенд для **сравнения алгоритмов решения задачи о назначениях** (assignment problem) по времени выполнения и качеству решения при фиксированном бюджете времени.
Внутри — три реализации:

* **ACO (муравьиный алгоритм)** — умеет работать с `--time_limit` и отдаёт лучшее найденное решение.
* **Венгерский (ручной)** — классический Мункрес; тоже «anytime» при `--time_limit`.
* **Венгерский (SciPy)** — библиотечный вариант алгоритма Венгерского. 

Есть **один единый бенчмарк**, который умеет запускать любой один алгоритм (с графиками) или все сразу, и складывает результаты в `out/`.

---

## Структура проекта

```
ant_colony/             # Муравьиный алгоритм
  aco.py                # AntColonyAssignment, ACOParams
  run.py                # CLI для одиночного запуска ACO
  utils.py              # Генерация матриц, greedy, brute-force для tiny, сохранение истории

hungarian_manual/       # Венгерский
  hungarian.py
  run.py

hungarian_scipy/        # Венгерский из SciPy
  hungarian_lib.py
  run.py

bench/
  benchmark_unified.py  # Единый бенчмарк: «один алгоритм» или «все сразу»

data/                   # Примеры матриц (CSV)
tests/                  # Санити-тесты на маленьких размерах
out/                    # ЕДИНЫЙ каталог результата (CSV/PNG)
```

---

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

Быстрый чек на tiny-матрице:

```bash
python -m ant_colony.run --csv data/tiny_4x4_diagonal.csv
python -m hungarian_scipy.run --csv data/tiny_4x4_diagonal.csv
python -m hungarian_manual.run --csv data/tiny_4x4_diagonal.csv
```

---

## Аргументы и запуск КАЖДОЙ реализации

### 1) ACO (муравьиный)

**Где:** `ant_colony/run.py` → `python -m ant_colony.run`

**Аргументы:**

* `--n` — размер квадратной матрицы (если не используете `--csv`).
* `--mode` — генератор: `uniform | normal | diag_dom | block`.
* `--csv` — путь к CSV-матрице (тогда генератор не нужен).
* `--seed` — зерно для воспроизводимости.
* `--time_limit` — бюджет времени (секунды). Включает режим **anytime**.
* `--history_csv` — если указать, сохраняет поитерационную историю ACO (CSV).
* `--ants` — число муравьёв на итерацию.
* `--alpha`, `--beta` — веса феромона и эвристики.
* `--rho` — испарение феромона.
* `--q0` — вероятность жадного шага (ACS-правило).
* `--Q` — масштаб депозита феромона.
* `--iters` — максимум итераций (если нет `--time_limit`).
* `--ls_iters` — число попыток локального поиска 2-swap.
* `--local_search` — `""` или `"2swap"`.
* `--verbose` — печатать прогресс.

**Примеры:**

```bash
# ACO c генерацией матрицы 120×120, бюджет 1.5 c, сохраняем историю
python -m ant_colony.run --n 120 --mode uniform --time_limit 1.5 --seed 42 --history_csv out/history_aco_120.csv

# ACO на своей CSV-матрице
python -m ant_colony.run --csv data/tiny_4x4_diagonal.csv --time_limit 0.2
```

---

### 2) Венгерский

**Где:** `hungarian_manual/run.py` → `python -m hungarian_manual.run`

**Аргументы:**

* `--n` — размер матрицы (если не `--csv`).
* `--csv` — путь к CSV-матрице (квадратная).
* `--seed` — зерно генерации (если используете генератор).
* `--time_limit` — бюджет времени. При таймауте возвращает **допустимое** приближённое решение (anytime).

**Примеры:**

```bash
# Генерация 300×300, жёсткий лимит времени
python -m hungarian_manual.run --n 300 --time_limit 0.2 --seed 7

# На CSV-матрице
python -m hungarian_manual.run --csv data/tiny_4x4_diagonal.csv
```

---

### 3) Венгерский (SciPy)

**Где:** `hungarian_scipy/run.py` → `python -m hungarian_scipy.run`

**Аргументы:**

* `--n` — размер матрицы (если не `--csv`).
* `--csv` — путь к CSV-матрице.
* `--seed` — зерно (для генератора).

**Примеры:**

```bash
python -m hungarian_scipy.run --n 200 --seed 42
python -m hungarian_scipy.run --csv data/tiny_4x4_diagonal.csv
```
---

## Единый бенчмарк

**Где:** `bench/benchmark_unified.py` → `python -m bench.benchmark_unified`

Назначение: одним интерфейсом запустить **любой один** алгоритм с полным отчётом и графиками, либо **все сразу**; сохранить **сводную таблицу** (`out/summary.csv`) и графики в `out/`.

### Аргументы

**Что запускать**

* `--algo` — `aco | hungarian_manual | hungarian_scipy | greedy | all` (по умолчанию `all`).

**Входные данные**

* `--n` — размер (если не `--sizes`/`--csv`).
* `--sizes` — несколько размеров через запятую, например: `100,200,300`.
* `--mode` — генератор матрицы: `uniform | normal | diag_dom | block`.
* `--csv` — путь к CSV-матрице (определяет `n` автоматически).
* `--seed` — зерно.

**Ограничение времени**

* `--time_limit` — общий бюджет времени.
  Это как раз сценарий «зафиксировать вычислительные мощности и время». SciPy запускается без лимита и даёт `opt_cost`.

**Параметры ACO** (если `--algo aco` или `--algo all`)

* `--ants --alpha --beta --rho --q0 --Q --iters --ls_iters --local_search --verbose`

**Куда складывать**

* `--outdir` — каталог результатов (по умолчанию `out`).
* `--out_csv` — путь к сводной таблице (по умолчанию `out/summary.csv`).
* `--history_csv` — сохранить историю ACO (если ACO запускался).

### Что сохраняется

* `out/summary.csv` — строки вида:

  ```
  algo,n,mode,seed,time_limit,status,cost,opt_cost,rel_error,elapsed,iterations
  ```

  где:

  * `opt_cost` — оптимум от SciPy,
  * `rel_error = (cost - opt_cost) / opt_cost`,
  * `elapsed` — фактическое время работы (для ACO/ручного учитывает `--time_limit`).
* Графики:

  * `out/bars_elapsed_n{n}.png` — столбики времени по алгоритмам для данного `n`;
  * `out/bars_rel_error_n{n}.png` — столбики относительной ошибки;
  * `out/convergence_aco_n{n}.png` — сходимость ACO (Best/Mean по итерациям; если ACO запускался);
  * при серии размеров: `out/lines_elapsed_vs_n.png`, `out/lines_rel_error_vs_n.png`.

### Примеры запуска

**Все алгоритмы на одном размере (фикс времени одинаковый для ACO и manual):**

```bash
python -m bench.benchmark_unified --algo all --n 200 --time_limit 1.0 --mode uniform --seed 42 --outdir out
```

**Серия размеров + общие графики:**

```bash
python -m bench.benchmark_unified --algo all --sizes 100,200,300,400 --time_limit 0.2 --mode uniform --seed 42 --outdir out
```

**Только ACO, с историей и конвергенцией:**

```bash
python -m bench.benchmark_unified --algo aco --n 200 --time_limit 1.0 --seed 42 --ants 80 --iters 4000 --local_search "2swap" --outdir out --history_csv out/history_aco_200.csv
```

**На своей CSV-матрице:**

```bash
python -m bench.benchmark_unified --algo all --csv data/tiny_4x4_diagonal.csv --time_limit 0.05 --outdir out
```

# Сравнение алгоритмов с оптимизированным АСО и без него. 

```
python run_compare_modes.py --sizes 25,50,100 --time_A 100 --time_B 0.5 --outroot out_3

python run_compare_modes_aco_opt.py --sizes 25,50,100,300,500,1000 --time_A 250 --time_B 1 --outroot out_1111_opt
```