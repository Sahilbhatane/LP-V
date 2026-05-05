"""
Sequential BFS and DFS on an undirected graph (same stdin layout as bfs_dfs.cpp).
Vertices are 0 .. V-1; invalid edges are rejected with an error message.

Run: python3 bfs_dfs.py
"""

from __future__ import annotations

from collections import deque
import sys
from typing import List


def bfs(adj: List[List[int]], start: int) -> List[int]:
    n = len(adj)
    visited = [False] * n
    order: List[int] = []
    q = deque([start])
    visited[start] = True

    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    return order


def dfs(adj: List[List[int]], start: int) -> List[int]:
    visited = [False] * len(adj)
    order: List[int] = []
    stack = [start]

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in reversed(adj[u]):
            if not visited[v]:
                stack.append(v)
    return order


def main() -> None:
    v = int(input("Enter number of vertices: ").strip())
    if v <= 0:
        print("Invalid vertex count.")
        sys.exit(1)

    adj: List[List[int]] = [[] for _ in range(v)]

    e = int(input("Enter number of edges: ").strip())
    if e < 0:
        print("Invalid edge count.")
        sys.exit(1)

    print("Enter edges (u v):")
    for _ in range(e):
        line = input().strip().split()
        if len(line) != 2:
            print("Expected two integers per line.")
            sys.exit(1)
        u, w = int(line[0]), int(line[1])
        if not (0 <= u < v and 0 <= w < v):
            print(f"Edge ({u},{w}) out of range 0..{v - 1}")
            sys.exit(1)
        if u != w:
            adj[u].append(w)
            adj[w].append(u)

    start = int(input("Enter starting vertex: ").strip())
    if not (0 <= start < v):
        print("Invalid start vertex.")
        sys.exit(1)

    print("\nBFS order:", " ".join(str(x) for x in bfs(adj, start)))
    print("DFS order:", " ".join(str(x) for x in dfs(adj, start)))


if __name__ == "__main__":
    main()
