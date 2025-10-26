from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np, time


@dataclass
class HungarianResult:
    perm: np.ndarray
    cost: float
    status: str
    elapsed: float
    iterations: int


def _complete_greedy(C0: np.ndarray, starred: np.ndarray) -> np.ndarray:
    """
    Ваш исходный достройщик: оставляю без изменений.
    Используется, если сработал таймаут (или нужно дополнить частичное сопоставление).
    """
    n = C0.shape[0]
    perm = -np.ones(n, dtype=int)
    used_cols = set()

    for i in range(n):
        js = np.where(starred[i])[0]
        if js.size > 0:
            j = int(js[0])
            perm[i] = j
            used_cols.add(j)

    for i in range(n):
        if perm[i] != -1:
            continue
        candidates = [j for j in range(n) if j not in used_cols]
        if not candidates:
            candidates = list(range(n))
        jbest = min(candidates, key=lambda j: C0[i, j])
        perm[i] = jbest
        used_cols.add(jbest)
    return perm


def hungarian_solve(C: np.ndarray, time_limit: Optional[float] = None) -> HungarianResult:
    """
    Исправленная и быстрая версия: классический венгерский с потенциалами.
    - Минимизирует сумму C[i, perm[i]] для квадратной матрицы C.
    - O(n^3), хорошо работает на float.
    - time_limit: при превышении возвращаем текущий частичный матчинг,
      дополненный _complete_greedy (как и раньше), status="timeout".
    """
    t0 = time.time()
    deadline = (t0 + time_limit) if (time_limit is not None) else None

    C0 = np.asarray(C, dtype=float)
    assert C0.ndim == 2 and C0.shape[0] == C0.shape[1], "C must be a square matrix"
    n = C0.shape[0]

    # Массивы потенциалов и паросочетания в форме «j-колонка -> p[j]-строка»
    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(n + 1, dtype=float)
    p = np.zeros(n + 1, dtype=int)  # p[0] – техническая вершина
    way = np.zeros(n + 1, dtype=int)

    iterations = 0  # считаем шаги внутреннего цикла

    # Основной цикл по строкам (1..n) — добавляем по одной строке в матчинге
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)  # текущие «расстояния» до столбцов
        used = np.zeros(n + 1, dtype=bool)

        while True:
            if deadline is not None and time.time() >= deadline:
                # Строим "starred" из текущего паросочетания p и достраиваем жадно
                starred = np.zeros((n, n), dtype=bool)
                for j in range(1, n + 1):
                    if p[j] != 0:
                        starred[p[j] - 1, j - 1] = True
                perm = _complete_greedy(C0, starred)
                cost = float(C0[np.arange(n), perm].sum())
                return HungarianResult(
                    perm=perm, cost=cost, status="timeout",
                    elapsed=(time.time() - t0), iterations=iterations
                )

            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            # Релаксация "кратчайших путей" в двойственных координатах
            Ci = C0[i0 - 1]  # строка i0 (0-индексация в C0)
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = Ci[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            # Подтягиваем потенциалы
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            iterations += 1
            if p[j0] == 0:
                break

        # Восстанавливаем увеличивающий путь и наращиваем паросочетание
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Строим итоговую перестановку из p: p[j] = i  => perm[i-1] = j-1
    perm = -np.ones(n, dtype=int)
    for j in range(1, n + 1):
        if p[j] > 0:
            perm[p[j] - 1] = j - 1

    # На всякий случай: если вдруг где-то остались -1 (не должно) — достроим жадно
    if (perm < 0).any():
        starred = np.zeros((n, n), dtype=bool)
        for i in range(n):
            if perm[i] >= 0:
                starred[i, perm[i]] = True
        perm = _complete_greedy(C0, starred)

    cost = float(C0[np.arange(n), perm].sum())
    return HungarianResult(
        perm=perm, cost=cost, status="ok",
        elapsed=(time.time() - t0), iterations=iterations
    )


