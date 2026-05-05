"""
Demonstrate parallel reductions for min, max, sum, and average.

Sequential reductions are implemented with plain Python loops. Parallel versions
split the array into contiguous chunks and aggregate partial results with a
multiprocessing pool. Install NumPy only if you extend this script with vector
variants later.

Usage:
    python3 reduce.py
"""

from __future__ import annotations

import multiprocessing as mp
import random
import time
from typing import Iterable, List, Tuple


def sequential_stats(values: Iterable[float]) -> Tuple[float, float, float, float]:
    """
    Compute minimum, maximum, sum, and arithmetic mean in one sequential pass.

    Args:
        values: Numeric iterable.

    Returns:
        Tuple (minimum, maximum, sum, average).
    """
    iterator = iter(values)
    try:
        first = float(next(iterator))
    except StopIteration:
        raise ValueError("Cannot summarise an empty sequence.") from None

    smallest = first
    largest = first
    total = first
    count = 1

    for item in iterator:
        number = float(item)
        if number < smallest:
            smallest = number
        if number > largest:
            largest = number
        total += number
        count += 1

    average = total / count
    return smallest, largest, total, average


def chunk_partial_stats(chunk: List[float]) -> Tuple[float, float, float, int]:
    """
    Reduce one slice of data into local min, max, sum, and element count.

    Args:
        chunk: Non-empty list slice owned by a worker process.

    Returns:
        Tuple (local_min, local_max, local_sum, local_count).
    """
    smallest = min(chunk)
    largest = max(chunk)
    total = sum(chunk)
    length = len(chunk)
    return smallest, largest, total, length


def parallel_stats(values: List[float], workers: int) -> Tuple[float, float, float, float]:
    """
    Parallel reduction by merging chunk-level statistics.

    Args:
        values: Full list of numbers.
        workers: Desired worker processes.

    Returns:
        Tuple (minimum, maximum, sum, average) consistent with sequential_stats.
    """
    if not values:
        raise ValueError("Cannot summarise an empty sequence.")

    worker_count = max(1, min(workers, len(values)))
    chunk_size = (len(values) + worker_count - 1) // worker_count
    chunks = [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]

    with mp.Pool(processes=len(chunks)) as pool:
        partial_results = pool.map(chunk_partial_stats, chunks)

    smallest = partial_results[0][0]
    largest = partial_results[0][1]
    total = partial_results[0][2]
    count = partial_results[0][3]

    for local_min, local_max, local_sum, local_count in partial_results[1:]:
        if local_min < smallest:
            smallest = local_min
        if local_max > largest:
            largest = local_max
        total += local_sum
        count += local_count

    average = total / count
    return smallest, largest, total, average


def main() -> None:
    """
    Ask how many samples to draw, compare sequential versus parallel reductions.
    """
    print("Parallel reduction demo (min, max, sum, average).")
    sample_count = int(input("How many random floats should we generate? ").strip())
    if sample_count <= 0:
        print("Sample count must be positive.")
        return

    workers = int(input("How many parallel workers should we use? ").strip())
    workers = max(1, workers)

    seed_text = input("Random seed (leave blank for nondeterministic): ").strip()
    if seed_text:
        random.seed(int(seed_text))

    data = [random.uniform(-1000.0, 1000.0) for _ in range(sample_count)]

    print("\nSequential reduction")
    start = time.perf_counter()
    seq_min, seq_max, seq_sum, seq_avg = sequential_stats(data)
    end = time.perf_counter()
    print("Seconds:", round(end - start, 8))
    print("Min:", seq_min, "Max:", seq_max, "Sum:", seq_sum, "Average:", seq_avg)

    print("\nParallel reduction")
    start_b = time.perf_counter()
    par_min, par_max, par_sum, par_avg = parallel_stats(data, workers)
    end_b = time.perf_counter()
    print("Seconds:", round(end_b - start_b, 8))
    print("Min:", par_min, "Max:", par_max, "Sum:", par_sum, "Average:", par_avg)

    checks = (
        abs(seq_min - par_min) < 1e-9,
        abs(seq_max - par_max) < 1e-9,
        abs(seq_sum - par_sum) < 1e-6,
        abs(seq_avg - par_avg) < 1e-9,
    )
    print("\nOutputs match within tolerance:", all(checks))


if __name__ == "__main__":
    mp.freeze_support()
    main()
