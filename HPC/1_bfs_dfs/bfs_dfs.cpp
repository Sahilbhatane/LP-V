/**
 * Parallel BFS and DFS (OpenMP) — compact SPPU-style lab version.
 *
 * Build: g++ -O2 -fopenmp -Wall -std=c++17 -o bfs_dfs bfs_dfs.cpp
 */

#include <iostream>
#include <omp.h>
#include <queue>
#include <vector>

using namespace std;

class Graph {
  int V;
  vector<vector<int>> adj;

public:
  explicit Graph(int vertices) : V(vertices) { adj.resize(V); }

  void addEdge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
  }

  void parallelBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;

    visited[start] = true;
    q.push(start);

    cout << "\nParallel BFS Traversal: ";

    while (!q.empty()) {
      int size = static_cast<int>(q.size());

#pragma omp parallel for
      for (int i = 0; i < size; i++) {
        int node = -1;

#pragma omp critical
        {
          if (!q.empty()) {
            node = q.front();
            q.pop();
            cout << node << " ";
          }
        }

        if (node != -1) {
          for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
#pragma omp critical
              {
                if (!visited[neighbor]) {
                  visited[neighbor] = true;
                  q.push(neighbor);
                }
              }
            }
          }
        }
      }
    }
    cout << endl;
  }

  void parallelDFSUtil(int node, vector<bool> &visited) {
    bool alreadyVisited;

#pragma omp critical
    {
      alreadyVisited = visited[node];
      if (!visited[node]) {
        visited[node] = true;
        cout << node << " ";
      }
    }

    if (alreadyVisited)
      return;

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(adj[node].size()); i++) {
      int neighbor = adj[node][i];

      if (!visited[neighbor]) {
#pragma omp task
        parallelDFSUtil(neighbor, visited);
      }
    }
  }

  void parallelDFS(int start) {
    vector<bool> visited(V, false);

    cout << "\nParallel DFS Traversal: ";

#pragma omp parallel
    {
#pragma omp single
      { parallelDFSUtil(start, visited); }
    }

    cout << endl;
  }
};

int main() {
  int V, E;

  cout << "Enter number of vertices (e.g. 6): ";
  cin >> V;
  if (V <= 0) {
    cerr << "Invalid vertex count.\n";
    return 1;
  }

  Graph g(V);

  cout << "Enter number of edges (e.g. 5): ";
  cin >> E;
  if (E < 0) {
    cerr << "Invalid edge count.\n";
    return 1;
  }

  cout << "Enter edges (u v):\n";
  for (int i = 0; i < E; i++) {
    int u, v;
    cin >> u >> v;
    if (u < 0 || u >= V || v < 0 || v >= V) {
      cerr << "Edge (" << u << "," << v << ") out of range 0.." << (V - 1) << "\n";
      return 1;
    }
    g.addEdge(u, v);
  }

  int start;
  cout << "Enter starting vertex (e.g. 0): ";
  cin >> start;
  if (start < 0 || start >= V) {
    cerr << "Invalid start vertex.\n";
    return 1;
  }

  g.parallelBFS(start);
  g.parallelDFS(start);

  return 0;
}
