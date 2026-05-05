"""
Parallel bubble sort and parallel merge sort with timing comparisons.

Sequential versions are included for measuring speedups. Python cannot apply
OpenMP pragmas directly; bubble phases use threads while merge chunks rely on
multiprocessing so you can compare wall-clock times against sequential runs.

Usage on Ubuntu:
    python3 par_sort.py
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


def sequential_bubble_sort(values: List[float]) -> List[float]:
    """
    Classic bubble sort with early exit flag.

    Args:
        values: Mutable sequence of comparable floats.

    Returns:
        New sorted list (input is not mutated).
    """
    arr = list(values)
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def parallel_bubble_sort(values: List[float], workers: int) -> List[float]:
    """
    Odd-even (brick) sort using threads so index swaps share one array safely.

    Args:
        values: Numbers to sort.
        workers: Thread pool size for parallel phases.

    Returns:
        Sorted copy of values.
    """
    arr = list(values)
    n = len(arr)
    if n <= 1:
        return arr

    def swap_pair(pair: Tuple[int, int]) -> None:
        """
        Compare neighbours at the given indices and swap out of order elements.
        """
        left_index, right_index = pair
        if arr[left_index] > arr[right_index]:
            arr[left_index], arr[right_index] = arr[right_index], arr[left_index]

    worker_cap = max(1, workers)
    with ThreadPoolExecutor(max_workers=worker_cap) as pool:
        for phase in range(n):
            odd_phase = phase % 2
            pairs: List[Tuple[int, int]] = []
            index = odd_phase
            while index < n - 1:
                pairs.append((index, index + 1))
                index += 2

            if pairs:
                list(pool.map(swap_pair, pairs))

            ordered = True
            for index in range(n - 1):
                if arr[index] > arr[index + 1]:
                    ordered = False
                    break
            if ordered:
                break

    return arr


def sequential_merge_sort(values: List[float]) -> List[float]:
    """
    Standard recursive merge sort returning a new sorted list.

    Args:
        values: Sequence of floats.

    Returns:
        Sorted list.
    """
    if len(values) <= 1:
        return list(values)

    mid = len(values) // 2
    left = sequential_merge_sort(values[:mid])
    right = sequential_merge_sort(values[mid:])
    return merge_two_sorted(left, right)


def merge_two_sorted(left: List[float], right: List[float]) -> List[float]:
    """
    Merge two sorted arrays into one sorted array.

    Args:
        left: Sorted ascending.
        right: Sorted ascending.

    Returns:
        Combined sorted list.
    """
    result: List[float] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    while i < len(left):
        result.append(left[i])
        i += 1
    while j < len(right):
        result.append(right[j])
        j += 1
    return result


def _merge_sort_worker(chunk: List[float]) -> List[float]:
    """
    Helper executed in child processes for parallel merge sort base case.

    Args:
        chunk: Sub-array owned by this worker.

    Returns:
        Sorted chunk.
    """
    return sequential_merge_sort(chunk)


def parallel_merge_sort(values: List[float], workers: int) -> List[float]:
    """
    Split the input across workers, merge-sort each chunk, then merge pairwise.

    Args:
        values: Numbers to sort.
        workers: Number of parallel partitions.

    Returns:
        Sorted list.
    """
    n = len(values)
    if n <= 1:
        return list(values)

    workers = max(1, min(workers, n))
    chunk_size = (n + workers - 1) // workers
    chunks = [values[i : i + chunk_size] for i in range(0, n, chunk_size)]

    with mp.Pool(processes=len(chunks)) as pool:
        sorted_chunks = pool.map(_merge_sort_worker, chunks)

    result = sorted_chunks[0]
    for part in sorted_chunks[1:]:
        result = merge_two_sorted(result, part)
    return result


def main() -> None:
    """
    Ask for array size and worker count, then print timings for each algorithm.
    """
    print("Parallel sorting benchmark (bubble and merge).")
    size = int(input("How many random floats should we sort (for example 2000)? ").strip())
    workers = int(input("How many parallel workers should we use? ").strip())
    workers = max(1, workers)

    seed_text = input("Random seed (integer, blank uses system default): ").strip()
    if seed_text:
        random.seed(int(seed_text))

    data = [random.random() for _ in range(size)]

    print("\nBubble sort — sequential")
    t0 = time.perf_counter()
    bub_seq = sequential_bubble_sort(data)
    t1 = time.perf_counter()
    print("Seconds:", round(t1 - t0, 6))

    print("\nBubble sort — parallel phases")
    t2 = time.perf_counter()
    bub_par = parallel_bubble_sort(data, workers)
    t3 = time.perf_counter()
    print("Seconds:", round(t3 - t2, 6))

    print("\nMerge sort — sequential")
    t4 = time.perf_counter()
    mer_seq = sequential_merge_sort(data)
    t5 = time.perf_counter()
    print("Seconds:", round(t5 - t4, 6))

    print("\nMerge sort — parallel chunks")
    t6 = time.perf_counter()
    mer_par = parallel_merge_sort(data, workers)
    t7 = time.perf_counter()
    print("Seconds:", round(t7 - t6, 6))

    ok_bubble = bub_seq == bub_par == sorted(data)
    ok_merge = mer_seq == mer_par == sorted(data)
    print("\nBubble outputs agree with sorted reference:", ok_bubble)
    print("Merge outputs agree with sorted reference:", ok_merge)


if __name__ == "__main__":
    mp.freeze_support()
    main()
