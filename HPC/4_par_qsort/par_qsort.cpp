/**
 * Sequential vs parallel quicksort (OpenMP tasks). Fixed size, no stdin.
 *
 * Build: g++ -O2 -fopenmp -Wall -std=c++17 -o par_qsort par_qsort.cpp
 */

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

static const int N = 5000;
static const size_t CUTOFF = 64;

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

static void quicksort_par_impl(vector<int> &data, size_t low, size_t high) {
  if (low >= high)
    return;
  if (high - low + 1 <= CUTOFF) {
    quicksort_seq(data, low, high);
    return;
  }
  size_t p = partition_vec(data, low, high);

#pragma omp task shared(data) if (p > low + 1)
  {
    if (p > low)
      quicksort_par_impl(data, low, p - 1);
  }
#pragma omp task shared(data) if (p + 1 < high)
  {
    if (p < high)
      quicksort_par_impl(data, p + 1, high);
  }
#pragma omp taskwait
}

int main() {
  vector<int> a(N);
  srand(static_cast<unsigned>(time(nullptr)));
  for (int &x : a)
    x = rand() % 100000;

  vector<int> seq = a;
  double t0 = omp_get_wtime();
  if (!seq.empty())
    quicksort_seq(seq, 0, seq.size() - 1);
  double t1 = omp_get_wtime();
  cout << "Sequential quicksort time: " << (t1 - t0) << " sec\n";

  vector<int> par = a;
  t0 = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single
    quicksort_par_impl(par, 0, par.size() - 1);
  }
  t1 = omp_get_wtime();
  cout << "Parallel quicksort time: " << (t1 - t0) << " sec\n";

  cout << "Sequential sorted: " << (is_sorted(seq.begin(), seq.end()) ? "yes" : "no") << "\n";
  cout << "Parallel sorted: " << (is_sorted(par.begin(), par.end()) ? "yes" : "no") << "\n";

  return 0;
}
