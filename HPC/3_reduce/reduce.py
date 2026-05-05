"""
Parallel reductions on a fixed array {1,2,3,4,5} (stdlib multiprocessing).

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
    arr = [1, 2, 3, 4, 5]
    n = len(arr)

    print("Sequential:")
    print("  minimum:", min(arr))
    print("  maximum:", max(arr))
    print("  sum:", sum(arr))
    print("  average:", sum(arr) / n)

    mn, mx, s = parallel_reduce(arr, 2)
    print("Parallel (chunk merge):")
    print("  minimum:", mn)
    print("  maximum:", mx)
    print("  sum:", s)
    print("  average:", s / n)


if __name__ == "__main__":
    main()
