"""
Sequential quicksort versus a parallel quicksort-style algorithm.

The parallel branch partitions once at the top level, sorts the left and right
blocks inside worker processes, then stitches the pieces together. That mirrors
the fork-join behaviour produced by OpenMP tasks in the C++ companion file.

Usage:
    python3 par_qsort.py
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from typing import List, Tuple


def sequential_quicksort(values: List[float]) -> List[float]:
    """
    Classic quicksort returning a new sorted list without mutating the argument.

    Args:
        values: Numbers to order.

    Returns:
        Sorted ascending list.
    """
    arr = list(values)

    def partition(low: int, high: int) -> int:
        """
        Lomuto partition scheme around the element at index high.

        Args:
            low: Left index inclusive.
            high: Right index inclusive.

        Returns:
            Final pivot index after swapping.
        """
        pivot_value = arr[high]
        smaller_zone = low - 1
        for walker in range(low, high):
            if arr[walker] <= pivot_value:
                smaller_zone += 1
                arr[smaller_zone], arr[walker] = arr[walker], arr[smaller_zone]
        arr[smaller_zone + 1], arr[high] = arr[high], arr[smaller_zone + 1]
        return smaller_zone + 1

    def recurse(low: int, high: int) -> None:
        """
        Recursive quicksort driver.

        Args:
            low: Left index inclusive.
            high: Right index inclusive.
        """
        if low < high:
            pivot_index = partition(low, high)
            recurse(low, pivot_index - 1)
            recurse(pivot_index + 1, high)

    if len(arr) <= 1:
        return arr
    recurse(0, len(arr) - 1)
    return arr


def partition_copy(values: List[float]) -> Tuple[List[float], float, List[float]]:
    """
    Partition a fresh list copy using Lomuto rules.

    Args:
        values: Full list of numbers.

    Returns:
        Tuple (elements less or equal than pivot, pivot value, elements greater).
    """
    arr = list(values)
    low = 0
    high = len(arr) - 1
    pivot_value = arr[high]
    smaller_zone = low - 1
    for walker in range(low, high):
        if arr[walker] <= pivot_value:
            smaller_zone += 1
            arr[smaller_zone], arr[walker] = arr[walker], arr[smaller_zone]
    arr[smaller_zone + 1], arr[high] = arr[high], arr[smaller_zone + 1]
    pivot_index = smaller_zone + 1
    left_part = arr[:pivot_index]
    pivot_final = arr[pivot_index]
    right_part = arr[pivot_index + 1 :]
    return left_part, pivot_final, right_part


def parallel_quicksort(values: List[float], threshold: int, workers: int) -> List[float]:
    """
    Hybrid quicksort: recurse sequentially on small arrays, otherwise fan out once.

    Args:
        values: Numbers to sort.
        threshold: Fall back to sequential quicksort when lengths fall below this.
        workers: Maximum worker processes for parallel segments.

    Returns:
        Sorted list.
    """
    if len(values) <= threshold:
        return sequential_quicksort(values)

    left_block, pivot_value, right_block = partition_copy(values)
    worker_cap = max(1, workers)

    chunks = []
    if len(left_block) > 0:
        chunks.append(left_block)
    if len(right_block) > 0:
        chunks.append(right_block)

    if not chunks:
        return [pivot_value]

    if len(chunks) == 1:
        single_name = chunks[0]
        return parallel_quicksort(single_name, threshold, worker_cap)

    process_count = min(worker_cap, len(chunks))
    with mp.Pool(processes=process_count) as pool:
        # Workers must only run sequential quicksort: nested Pool() inside a pool
        # worker raises AssertionError on Windows ("daemonic processes may not spawn").
        sorted_chunks = pool.map(sort_worker, chunks)

    if len(sorted_chunks) == 2:
        return sorted_chunks[0] + [pivot_value] + sorted_chunks[1]
    return sorted_chunks[0]


def sort_worker(chunk: List[float]) -> List[float]:
    """
    Sort one partition using the sequential routine so child processes stay leaf workers.

    Args:
        chunk: Sub-array produced by the parent partition step.

    Returns:
        Sorted chunk.
    """
    return sequential_quicksort(chunk)


def main() -> None:
    """
    Gather parameters from stdin and display sequential versus parallel timings.
    """
    print("Parallel quicksort benchmark.")
    size = int(input("How many random floats should we sort? ").strip())
    threshold = int(
        input(
            "Fall back to sequential sort when subproblem size is below "
            "(for example 8000): "
        ).strip()
    )
    workers = int(input("Maximum parallel worker processes: ").strip())
    workers = max(1, workers)

    seed_text = input("Random seed (optional integer): ").strip()
    if seed_text:
        random.seed(int(seed_text))

    data = [random.random() for _ in range(size)]

    print("\nSequential quicksort")
    start = time.perf_counter()
    seq_result = sequential_quicksort(data)
    end = time.perf_counter()
    print("Seconds:", round(end - start, 6))

    print("\nParallel quicksort")
    start_b = time.perf_counter()
    par_result = parallel_quicksort(list(data), threshold, workers)
    end_b = time.perf_counter()
    print("Seconds:", round(end_b - start_b, 6))

    reference = sorted(data)
    print("\nSequential matches sorted():", seq_result == reference)
    print("Parallel matches sorted():", par_result == reference)


if __name__ == "__main__":
    mp.freeze_support()
    main()
