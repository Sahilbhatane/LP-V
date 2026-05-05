/**
 * Parallel BFS and DFS on an undirected graph using OpenMP.
 *
 * Build on Ubuntu:
 *   g++ -O2 -fopenmp -Wall -std=c++17 -o bfs_dfs bfs_dfs.cpp
 *
 * Run:
 *   ./bfs_dfs
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

using namespace std;

/**
 * Read graph size and edges from stdin.
 */
static void read_graph(int &n, int &m, vector<pair<int, int>> &edges) {
  cout << "Number of vertices: ";
  cin >> n;
  cout << "Number of undirected edges: ";
  cin >> m;
  edges.resize(m);
  for (int i = 0; i < m; ++i) {
    int u, v;
    cout << "Edge " << (i + 1) << " (u v): ";
    cin >> u >> v;
    edges[i] = {u, v};
  }
}

/**
 * Build adjacency list for an undirected simple graph.
 */
static vector<vector<int>> build_adjacency(int n,
                                           const vector<pair<int, int>> &edges) {
  vector<vector<int>> adj(n);
  for (const auto &e : edges) {
    int u = e.first;
    int v = e.second;
    if (u == v)
      continue;
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  for (int i = 0; i < n; ++i) {
    sort(adj[i].begin(), adj[i].end());
    adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
  }
  return adj;
}

/**
 * Sequential BFS for baseline timing and reference order.
 */
static vector<int> bfs_sequential(const vector<vector<int>> &adj, int start) {
  int n = static_cast<int>(adj.size());
  vector<char> visited(n, 0);
  vector<int> order;
  queue<int> q;
  visited[start] = 1;
  q.push(start);

  while (!q.empty()) {
    int u = q.front();
    q.pop();
    order.push_back(u);
    for (int v : adj[u]) {
      if (!visited[v]) {
        visited[v] = 1;
        q.push(v);
      }
    }
  }
  return order;
}

/**
 * Parallel level-synchronous BFS using OpenMP over the current frontier.
 */
static vector<int> bfs_parallel(const vector<vector<int>> &adj, int start) {
  int n = static_cast<int>(adj.size());
  vector<char> visited(n, 0);
  vector<int> order;
  vector<int> frontier;
  frontier.push_back(start);
  visited[start] = 1;

  while (!frontier.empty()) {
    order.insert(order.end(), frontier.begin(), frontier.end());

    vector<int> local_candidates;
#pragma omp parallel
    {
      vector<int> thread_private;
#pragma omp for schedule(static) nowait
      for (long long ii = 0; ii < static_cast<long long>(frontier.size()); ++ii) {
        size_t i = static_cast<size_t>(ii);
        int u = frontier[i];
        for (int v : adj[u]) {
          thread_private.push_back(v);
        }
      }
#pragma omp critical
      {
        local_candidates.insert(local_candidates.end(), thread_private.begin(),
                                thread_private.end());
      }
    }

    vector<int> next_frontier;
    for (int v : local_candidates) {
      if (!visited[v]) {
        visited[v] = 1;
        next_frontier.push_back(v);
      }
    }
    sort(next_frontier.begin(), next_frontier.end());
    next_frontier.erase(unique(next_frontier.begin(), next_frontier.end()),
                        next_frontier.end());
    frontier.swap(next_frontier);
  }
  return order;
}

/**
 * Sequential DFS (preorder with stack).
 */
static vector<int> dfs_sequential(const vector<vector<int>> &adj, int start) {
  int n = static_cast<int>(adj.size());
  vector<char> visited(n, 0);
  vector<int> order;
  stack<int> st;
  st.push(start);

  while (!st.empty()) {
    int u = st.top();
    st.pop();
    if (visited[u])
      continue;
    visited[u] = 1;
    order.push_back(u);
    for (auto it = adj[u].rbegin(); it != adj[u].rend(); ++it) {
      int v = *it;
      if (!visited[v])
        st.push(v);
    }
  }
  return order;
}

/**
 * Depth-first exploration from one starting vertex using a stack.
 * Used inside parallel DFS worker branches (same pattern as the Python helper).
 */
static vector<int> dfs_from_vertex(const vector<vector<int>> &adj,
                                   vector<char> visited_seed, int entry) {
  int n = static_cast<int>(adj.size());
  vector<int> order_local;
  stack<int> st;
  st.push(entry);

  while (!st.empty()) {
    int u = st.top();
    st.pop();
    if (visited_seed[u])
      continue;
    visited_seed[u] = 1;
    order_local.push_back(u);
    for (auto it = adj[u].rbegin(); it != adj[u].rend(); ++it) {
      int v = *it;
      if (!visited_seed[v])
        st.push(v);
    }
  }
  return order_local;
}

/**
 * Parallel DFS variant: fix the root visit, then split sorted neighbours across threads.
 * Each thread carries a copy of the visited bitmap seeded with the root only,
 * matching the Python multiprocessing sketch for classroom experiments.
 */
static vector<int> dfs_parallel_split(const vector<vector<int>> &adj, int start) {
  int n = static_cast<int>(adj.size());
  vector<int> order;
  order.push_back(start);

  vector<int> children = adj[start];
  sort(children.begin(), children.end());

  if (children.empty()) {
    return order;
  }

  vector<vector<int>> branches(children.size());

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < static_cast<int>(children.size()); ++i) {
    vector<char> seed_vis(n, 0);
    seed_vis[start] = 1;
    branches[(size_t)i] = dfs_from_vertex(adj, seed_vis, children[(size_t)i]);
  }

  vector<char> seen(n, 0);
  seen[start] = 1;
  for (const auto &branch : branches) {
    for (int node : branch) {
      if (!seen[node]) {
        seen[node] = 1;
        order.push_back(node);
      }
    }
  }

  return order;
}

int main() {
  int n, m;
  vector<pair<int, int>> edges;
  read_graph(n, m, edges);

  int start;
  cout << "Start vertex: ";
  cin >> start;

  if (start < 0 || start >= n) {
    cerr << "Invalid start vertex.\n";
    return 1;
  }

  vector<vector<int>> adj = build_adjacency(n, edges);

  auto t0 = chrono::high_resolution_clock::now();
  vector<int> b_seq = bfs_sequential(adj, start);
  auto t1 = chrono::high_resolution_clock::now();

  auto t2 = chrono::high_resolution_clock::now();
  vector<int> b_par = bfs_parallel(adj, start);
  auto t3 = chrono::high_resolution_clock::now();

  cout << "\nSequential BFS order:";
  for (int x : b_seq)
    cout << ' ' << x;
  cout << "\nSequential BFS time (ms): "
       << chrono::duration<double, milli>(t1 - t0).count() << "\n";

  cout << "Parallel BFS order:";
  for (int x : b_par)
    cout << ' ' << x;
  cout << "\nParallel BFS time (ms): "
       << chrono::duration<double, milli>(t3 - t2).count() << "\n";

  auto t4 = chrono::high_resolution_clock::now();
  vector<int> d_seq = dfs_sequential(adj, start);
  auto t5 = chrono::high_resolution_clock::now();

  auto t6 = chrono::high_resolution_clock::now();
  vector<int> d_par = dfs_parallel_split(adj, start);
  auto t7 = chrono::high_resolution_clock::now();

  cout << "\nSequential DFS order:";
  for (int x : d_seq)
    cout << ' ' << x;
  cout << "\nSequential DFS time (ms): "
       << chrono::duration<double, milli>(t5 - t4).count() << "\n";

  cout << "Parallel DFS order (split children):";
  for (int x : d_par)
    cout << ' ' << x;
  cout << "\nParallel DFS time (ms): "
       << chrono::duration<double, milli>(t7 - t6).count() << "\n";

  return 0;
}
