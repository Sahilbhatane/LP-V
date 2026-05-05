/*
 * Parallel reductions (min, max, sum, average) with OpenMP. Reads array from stdin.
 *
 * Build: g++ -O2 -fopenmp -Wall -std=c++17 -o reduce reduce.cpp
 *
 * Note: Prefer GCC on Linux/WSL/MSYS2 for OpenMP reductions on int.
 */

#include <algorithm>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

int minval(int *arr, int n) {
  int mn = arr[0];
#pragma omp parallel for reduction(min : mn)
  for (int i = 0; i < n; i++) {
    if (arr[i] < mn)
      mn = arr[i];
  }
  return mn;
}

int maxval(int *arr, int n) {
  int mx = arr[0];
#pragma omp parallel for reduction(max : mx)
  for (int i = 0; i < n; i++) {
    if (arr[i] > mx)
      mx = arr[i];
  }
  return mx;
}

int sum_arr(int *arr, int n) {
  int s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < n; i++)
    s += arr[i];
  return s;
}

double average_from_sum(long long sum, int n) { return static_cast<double>(sum) / n; }

int main() {
  int n;
  cout << "How many integers? (e.g. 5): ";
  cin >> n;
  if (n <= 0) {
    cerr << "Count must be positive.\n";
    return 1;
  }

  vector<int> arr(static_cast<size_t>(n));
  cout << "Enter " << n << " integers, space-separated (e.g. 1 2 3 4 5): ";
  for (int i = 0; i < n; ++i)
    cin >> arr[i];

  int *p = arr.data();

  long long seq_sum = 0;
  int seq_min = p[0], seq_max = p[0];
  for (int i = 0; i < n; ++i) {
    seq_min = min(seq_min, p[i]);
    seq_max = max(seq_max, p[i]);
    seq_sum += p[i];
  }

  cout << "\nSequential min / max / sum / avg: " << seq_min << " / " << seq_max << " / " << seq_sum
       << " / " << average_from_sum(seq_sum, n) << "\n";

  int sm = sum_arr(p, n);
  cout << "Parallel min / max / sum / avg: " << minval(p, n) << " / " << maxval(p, n) << " / " << sm
       << " / " << average_from_sum(static_cast<long long>(sm), n) << "\n";

  return 0;
}
