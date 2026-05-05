"""
Parallel breadth-first and depth-first search on an undirected graph.

This script mirrors the ideas used in the OpenMP C++ companion program.
Pure Python does not expose OpenMP pragmas, so concurrency uses the
standard library multiprocessing pool to explore frontier nodes in BFS
and to spawn parallel DFS subtrees when you enable parallel mode.

Run on Ubuntu with Python 3:
    python3 bfs_dfs.py
"""

from __future__ import annotations

from collections import deque
from multiprocessing import Pool
import sys
import time
from typing import Dict, List, Set, Tuple


def build_undirected_graph(num_vertices: int, edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """
    Build an adjacency list for an undirected graph from edge pairs.

    Args:
        num_vertices: Number of vertices labelled from 0 to num_vertices - 1.
        edges: List of (u, v) pairs; each edge is stored twice for undirected behaviour.

    Returns:
        Mapping from vertex index to sorted list of neighbour indices.
    """
    graph: Dict[int, List[int]] = {i: [] for i in range(num_vertices)}
    for u, v in edges:
        if u == v:
            continue
        graph[u].append(v)
        graph[v].append(u)
    for key in graph:
        graph[key] = sorted(set(graph[key]))
    return graph


def sequential_bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Standard BFS producing vertices in visit order.

    Args:
        graph: Adjacency list representation.
        start: Source vertex.

    Returns:
        Order of first visits starting from start.
    """
    visited: Set[int] = set()
    order: List[int] = []
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)

    return order


def sequential_dfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Depth-first search using an explicit stack (preorder traversal).

    Args:
        graph: Adjacency list representation.
        start: Source vertex.

    Returns:
        DFS preorder list from start.
    """
    visited: Set[int] = set()
    order: List[int] = []
    stack = [start]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for neighbour in reversed(graph.get(node, [])):
            if neighbour not in visited:
                stack.append(neighbour)

    return order


def _expand_frontier_chunk(args: Tuple[List[int], Dict[int, List[int]], Set[int]]) -> List[int]:
    """
    Worker helper used by parallel BFS to collect neighbours for a chunk of the frontier.

    Args:
        args: Tuple of (chunk_of_frontier, graph_copy_serializable, visited_snapshot_ids).

    Returns:
        List of neighbour vertices discovered from this chunk (may contain duplicates).
    """
    chunk, graph_local, visited_snapshot = args
    out: List[int] = []
    for node in chunk:
        for neighbour in graph_local.get(node, []):
            if neighbour not in visited_snapshot:
                out.append(neighbour)
    return out


def parallel_bfs(
    graph: Dict[int, List[int]],
    start: int,
    workers: int,
) -> List[int]:
    """
    Level-synchronous parallel BFS using multiprocessing over frontier batches.

    Args:
        graph: Adjacency list representation.
        start: Source vertex.
        workers: Number of worker processes.

    Returns:
        Visit order similar to sequential BFS for the same tie-breaking rule.
    """
    visited: Set[int] = {start}
    order: List[int] = []
    frontier = [start]

    graph_tuple_friendly = {k: list(v) for k, v in graph.items()}

    while frontier:
        order.extend(frontier)

        if workers <= 1 or len(frontier) == 1:
            next_frontier: List[int] = []
            for node in frontier:
                for neighbour in graph_tuple_friendly.get(node, []):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.append(neighbour)
            frontier = sorted(set(next_frontier))
            continue

        chunk_size = max(1, (len(frontier) + workers - 1) // workers)
        chunks = [frontier[i : i + chunk_size] for i in range(0, len(frontier), chunk_size)]

        with Pool(processes=min(workers, len(chunks))) as pool:
            partial_lists = pool.map(
                _expand_frontier_chunk,
                [(ch, graph_tuple_friendly, visited) for ch in chunks],
            )

        candidates: List[int] = []
        for lst in partial_lists:
            candidates.extend(lst)

        next_set: Set[int] = set()
        for node in candidates:
            if node not in visited:
                visited.add(node)
                next_set.add(node)

        frontier = sorted(next_set)

    return order


def _dfs_branch(args: Tuple[int, Dict[int, List[int]], Tuple[int, ...]]) -> List[int]:
    """
    Explore one DFS branch starting from one child (used when parallel DFS is enabled).

    Args:
        args: Tuple of (child_vertex, graph_mapping, visited_tuple_so_far).

    Returns:
        DFS preorder for nodes reachable through child given frozen visited prefix.
    """
    root_child, graph_local, frozen = args
    visited: Set[int] = set(frozen)
    order_local: List[int] = []
    stack = [root_child]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order_local.append(node)
        for neighbour in reversed(graph_local.get(node, [])):
            if neighbour not in visited:
                stack.append(neighbour)

    return order_local


def parallel_dfs_first_level(
    graph: Dict[int, List[int]],
    start: int,
    workers: int,
) -> List[int]:
    """
    Parallel DFS variant: after visiting start, split children across worker processes.

    Args:
        graph: Adjacency list representation.
        start: Source vertex.
        workers: Number of worker processes.

    Returns:
        Combined preorder where start comes first, then subtrees merged by sorted child order.
    """
    graph_copy = {k: list(v) for k, v in graph.items()}
    children_start = list(graph_copy.get(start, []))

    if workers <= 1 or len(children_start) <= 1:
        return sequential_dfs(graph, start)

    order: List[int] = [start]
    visited_global: Set[int] = {start}

    frozen_tuple = tuple(sorted(visited_global))

    tasks = []
    for child in sorted(children_start):
        tasks.append((child, graph_copy, frozen_tuple))

    with Pool(processes=min(workers, len(tasks))) as pool:
        branches = pool.map(_dfs_branch, tasks)

    seen: Set[int] = set(order)
    for branch_list in branches:
        for node in branch_list:
            if node not in seen:
                seen.add(node)
                order.append(node)

    return order


def read_graph_from_user() -> Tuple[Dict[int, List[int]], int, int]:
    """
    Interactive prompts to build an undirected graph and choose BFS or DFS start.

    Returns:
        Tuple of (adjacency_list_graph, start_vertex, worker_count).
    """
    print("Build an undirected graph (vertices numbered from 0).")
    num_vertices = int(input("Number of vertices: ").strip())
    if num_vertices <= 0:
        print("Need at least one vertex.")
        sys.exit(1)

    num_edges = int(input("Number of undirected edges: ").strip())
    edges: List[Tuple[int, int]] = []
    for i in range(num_edges):
        line = input(f"Edge {i + 1} as two integers u v separated by space: ").strip().split()
        if len(line) != 2:
            print("Expected two integers per edge.")
            sys.exit(1)
        u, v = int(line[0]), int(line[1])
        edges.append((u, v))

    start = int(input("Start vertex for traversal: ").strip())
    if start < 0 or start >= num_vertices:
        print("Start vertex out of range.")
        sys.exit(1)

    workers = int(input("Number of parallel workers (1 = sequential style): ").strip())
    if workers < 1:
        workers = 1

    graph = build_undirected_graph(num_vertices, edges)
    return graph, start, workers


def main() -> None:
    """
    Entry point: run timed sequential and parallel traversals and print results.
    """
    graph, start, workers = read_graph_from_user()

    print("\n--- Breadth-first search ---")
    t0 = time.perf_counter()
    bfs_seq = sequential_bfs(graph, start)
    t1 = time.perf_counter()
    print("Sequential BFS order:", bfs_seq)
    print("Sequential BFS time (seconds):", round(t1 - t0, 6))

    t2 = time.perf_counter()
    bfs_par = parallel_bfs(graph, start, workers)
    t3 = time.perf_counter()
    print("Parallel BFS order:", bfs_par)
    print("Parallel BFS time (seconds):", round(t3 - t2, 6))

    print("\n--- Depth-first search ---")
    t4 = time.perf_counter()
    dfs_seq = sequential_dfs(graph, start)
    t5 = time.perf_counter()
    print("Sequential DFS order:", dfs_seq)
    print("Sequential DFS time (seconds):", round(t5 - t4, 6))

    t6 = time.perf_counter()
    dfs_par = parallel_dfs_first_level(graph, start, workers)
    t7 = time.perf_counter()
    print("Parallel DFS order (first-level split):", dfs_par)
    print("Parallel DFS time (seconds):", round(t7 - t6, 6))


if __name__ == "__main__":
    main()
