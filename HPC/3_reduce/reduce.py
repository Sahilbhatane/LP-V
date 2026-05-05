"""
Parallel reductions (min, max, sum, average). Reads count, numbers, and worker count from stdin.

Run: python3 reduce.py
"""

from __future__ import annotations

from multiprocessing import Pool


def stats_chunk(chunk: list[int]) -> tuple[int, int, int]:
    return min(chunk), max(chunk), sum(chunk)


def parallel_reduce(arr: list[int], workers: int) -> tuple[int, int, int]:
    n = len(arr)
    w = max(1, min(workers, n))
    step = (n + w - 1) // w
    chunks = [arr[i : i + step] for i in range(0, n, step)]
    with Pool(processes=len(chunks)) as pool:
        parts = pool.map(stats_chunk, chunks)
    mn = parts[0][0]
    mx = parts[0][1]
    s = parts[0][2]
    for a, b, c in parts[1:]:
        mn = min(mn, a)
        mx = max(mx, b)
        s += c
    return mn, mx, s


def main() -> None:
    n = int(input("How many integers? (e.g. 5): ").strip())
    if n <= 0:
        print("Count must be positive.")
        return

    line = input(
        f"Enter {n} integers, space-separated (e.g. 1 2 3 4 5): "
    ).strip().split()
    if len(line) != n:
        print(f"Expected {n} numbers, got {len(line)}.")
        return
    arr = [int(x) for x in line]

    workers = int(input("Worker processes for parallel reduction (e.g. 2): ").strip())
    workers = max(1, workers)

    seq_sum = sum(arr)
    print("\nSequential:")
    print("  minimum:", min(arr))
    print("  maximum:", max(arr))
    print("  sum:", seq_sum)
    print("  average:", seq_sum / n)

    mn, mx, s = parallel_reduce(arr, workers)
    print("Parallel (chunk merge):")
    print("  minimum:", mn)
    print("  maximum:", mx)
    print("  sum:", s)
    print("  average:", s / n)


if __name__ == "__main__":
    main()
