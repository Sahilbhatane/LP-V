"""
Sequential quicksort vs one-level parallel partition + Pool workers (fixed N).

Run: python3 par_qsort.py
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from typing import List

N = 5000


def quicksort_seq(values: List[int]) -> List[int]:
    if len(values) <= 1:
        return list(values)
    pivot = values[-1]
    left = [x for x in values[:-1] if x <= pivot]
    right = [x for x in values[:-1] if x > pivot]
    return quicksort_seq(left) + [pivot] + quicksort_seq(right)


def sort_worker(chunk: List[int]) -> List[int]:
    return quicksort_seq(chunk)


def quicksort_par_top(values: List[int], workers: int) -> List[int]:
    if len(values) <= 1:
        return list(values)
    pivot = values[-1]
    left = [x for x in values[:-1] if x <= pivot]
    right = [x for x in values[:-1] if x > pivot]
    parts: List[List[int]] = [left, right]
    w = min(max(1, workers), 2)
    with mp.Pool(w) as pool:
        sorted_parts = pool.map(sort_worker, parts)
    return sorted_parts[0] + [pivot] + sorted_parts[1]


def main() -> None:
    random.seed()
    data = [random.randint(0, 99999) for _ in range(N)]

    t0 = time.perf_counter()
    a = quicksort_seq(data)
    t1 = time.perf_counter()
    print("Sequential quicksort time:", round(t1 - t0, 6), "sec")

    t0 = time.perf_counter()
    b = quicksort_par_top(list(data), 2)
    t1 = time.perf_counter()
    print("Parallel quicksort time:", round(t1 - t0, 6), "sec")

    ref = sorted(data)
    print("Sequential correct:", a == ref)
    print("Parallel correct:", b == ref)


if __name__ == "__main__":
    mp.freeze_support()
    main()
