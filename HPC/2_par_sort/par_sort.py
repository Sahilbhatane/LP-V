"""
Parallel bubble + merge sort timings (stdlib only). Fixed SIZE (same idea as par_sort.cpp).

Run: python3 par_sort.py
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

SIZE = 256  # Python threaded odd-even has high overhead; keep small (C++ uses 10000)


def bubble_seq(arr: List[int]) -> None:
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]


def swap_pair(args: Tuple[List[int], int, int]) -> None:
    a, i, j = args
    if a[i] > a[j]:
        a[i], a[j] = a[j], a[i]


def bubble_par(arr: List[int]) -> None:
    n = len(arr)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in range(n):
            for phase in (0, 1):
                pairs = [(arr, k, k + 1) for k in range(phase, n - 1, 2)]
                if pairs:
                    list(pool.map(swap_pair, pairs))


def merge_two(left: List[int], right: List[int]) -> List[int]:
    out: List[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def merge_sort_seq(values: List[int]) -> List[int]:
    if len(values) <= 1:
        return list(values)
    m = len(values) // 2
    return merge_two(merge_sort_seq(values[:m]), merge_sort_seq(values[m:]))


def merge_sort_chunk(chunk: List[int]) -> List[int]:
    return merge_sort_seq(chunk)


def merge_sort_par(values: List[int], workers: int) -> List[int]:
    if len(values) <= 1:
        return list(values)
    n_workers = max(1, min(workers, len(values)))
    chunk_sz = (len(values) + n_workers - 1) // n_workers
    chunks = [values[i : i + chunk_sz] for i in range(0, len(values), chunk_sz)]
    with mp.Pool(processes=len(chunks)) as pool:
        parts = pool.map(merge_sort_chunk, chunks)
    out = parts[0]
    for p in parts[1:]:
        out = merge_two(out, p)
    return out


def main() -> None:
    random.seed()
    base = [random.randint(0, 99999) for _ in range(SIZE)]

    a = base[:]
    t0 = time.perf_counter()
    bubble_seq(a)
    t1 = time.perf_counter()
    print("Sequential Bubble Sort Time:", round(t1 - t0, 6), "sec")

    a = base[:]
    t0 = time.perf_counter()
    bubble_par(a)
    t1 = time.perf_counter()
    print("Parallel Bubble Sort Time:", round(t1 - t0, 6), "sec")

    t0 = time.perf_counter()
    _ = merge_sort_seq(base[:])
    t1 = time.perf_counter()
    print("Sequential Merge Sort Time:", round(t1 - t0, 6), "sec")

    t0 = time.perf_counter()
    _ = merge_sort_par(base[:], 4)
    t1 = time.perf_counter()
    print("Parallel Merge Sort Time:", round(t1 - t0, 6), "sec")


if __name__ == "__main__":
    mp.freeze_support()
    main()
