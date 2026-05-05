/**
 * Sequential vs parallel quicksort (OpenMP tasks). Reads size, cutoff, threads, seed from stdin.
 *
 * Build: g++ -O2 -fopenmp -Wall -std=c++17 -o par_qsort par_qsort.cpp
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

static size_t partition_vec(vector<int> &data, size_t low, size_t high) {
  int pivot = data[high];
  long long i = static_cast<long long>(low) - 1;
  for (long long j = static_cast<long long>(low); j <= static_cast<long long>(high) - 1; ++j) {
    if (data[static_cast<size_t>(j)] <= pivot) {
      ++i;
      swap(data[static_cast<size_t>(i)], data[static_cast<size_t>(j)]);
    }
  }
  swap(data[static_cast<size_t>(i + 1)], data[high]);
  return static_cast<size_t>(i + 1);
}

static void quicksort_seq(vector<int> &data, size_t low, size_t high) {
  if (low >= high)
    return;
  size_t p = partition_vec(data, low, high);
  if (p > 0)
    quicksort_seq(data, low, p - 1);
  if (p < high)
    quicksort_seq(data, p + 1, high);
}

static void quicksort_par_impl(vector<int> &data, size_t low, size_t high, size_t cutoff) {
  if (low >= high)
    return;
  if (high - low + 1 <= cutoff) {
    quicksort_seq(data, low, high);
    return;
  }
  size_t p = partition_vec(data, low, high);

#pragma omp task shared(data, cutoff) if (p > low + 1)
  {
    if (p > low)
      quicksort_par_impl(data, low, p - 1, cutoff);
  }
#pragma omp task shared(data, cutoff) if (p + 1 < high)
  {
    if (p < high)
      quicksort_par_impl(data, p + 1, high, cutoff);
  }
#pragma omp taskwait
}

static void run_parallel(vector<int> &data, size_t cutoff) {
  if (data.empty())
    return;
#pragma omp parallel
  {
#pragma omp single
    quicksort_par_impl(data, 0, data.size() - 1, cutoff);
  }
}

int main() {
  int n;
  cout << "How many random integers? (e.g. 5000): ";
  cin >> n;
  if (n <= 0) {
    cerr << "Size must be positive.\n";
    return 1;
  }

  size_t cutoff;
  cout << "Sequential cutoff for parallel recursion (e.g. 64; subarrays this size use seq. sort): ";
  cin >> cutoff;
  if (cutoff == 0)
    cutoff = 1;

  int threads;
  cout << "OpenMP threads (e.g. 4): ";
  cin >> threads;
  threads = max(1, threads);
  omp_set_num_threads(threads);

  unsigned seed;
  cout << "Random seed (e.g. 4242): ";
  cin >> seed;

  vector<int> a(static_cast<size_t>(n));
  srand(seed);
  for (int &x : a)
    x = rand() % 100000;

  vector<int> seq = a;
  double t0 = omp_get_wtime();
  if (!seq.empty())
    quicksort_seq(seq, 0, seq.size() - 1);
  double t1 = omp_get_wtime();
  cout << "\nSequential quicksort time: " << (t1 - t0) << " sec\n";

  vector<int> par = a;
  t0 = omp_get_wtime();
  run_parallel(par, cutoff);
  t1 = omp_get_wtime();
  cout << "Parallel quicksort time: " << (t1 - t0) << " sec\n";

  cout << "Sequential sorted: " << (is_sorted(seq.begin(), seq.end()) ? "yes" : "no") << "\n";
  cout << "Parallel sorted: " << (is_sorted(par.begin(), par.end()) ? "yes" : "no") << "\n";

  return 0;
}
