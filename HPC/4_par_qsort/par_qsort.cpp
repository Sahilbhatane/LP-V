/**
 * Sequential quicksort versus an OpenMP task-based parallel quicksort.
 *
 * Build:
 *   g++ -O2 -fopenmp -Wall -std=c++17 -o par_qsort par_qsort.cpp
 *
 * Run:
 *   ./par_qsort
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

using namespace std;

/**
 * Lomuto partition on the inclusive range [low, high].
 */
static size_t partition(vector<double> &data, size_t low, size_t high) {
  double pivot = data[high];
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

/**
 * Recursive sequential quicksort helper.
 */
static void quicksort_seq(vector<double> &data, size_t low, size_t high) {
  if (low >= high)
    return;
  size_t pivot_index = partition(data, low, high);
  if (pivot_index > 0)
    quicksort_seq(data, low, pivot_index - 1);
  if (pivot_index < high)
    quicksort_seq(data, pivot_index + 1, high);
}

/**
 * Wrapper around sequential quicksort for timing comparisons.
 */
static void run_sequential(vector<double> data) {
  if (!data.empty())
    quicksort_seq(data, 0, data.size() - 1);
}

/**
 * Recursive parallel quicksort using OpenMP tasks with a grain cutoff.
 */
static void quicksort_par_impl(vector<double> &data, size_t low, size_t high,
                               size_t cutoff) {
  if (low >= high)
    return;

  size_t span = high - low + 1;
  if (span <= cutoff) {
    quicksort_seq(data, low, high);
    return;
  }

  size_t pivot_index = partition(data, low, high);

#pragma omp task shared(data, cutoff) if (pivot_index > low + 1)
  {
    if (pivot_index > low)
      quicksort_par_impl(data, low, pivot_index - 1, cutoff);
  }

#pragma omp task shared(data, cutoff) if (pivot_index + 1 < high)
  {
    if (pivot_index < high)
      quicksort_par_impl(data, pivot_index + 1, high, cutoff);
  }

#pragma omp taskwait
}

static void run_parallel(vector<double> &data, size_t cutoff) {
  if (data.empty())
    return;
#pragma omp parallel
  {
#pragma omp single
    quicksort_par_impl(data, 0, data.size() - 1, cutoff);
  }
}

/**
 * Fill vector with reproducible pseudo-random doubles.
 */
static void fill_random(vector<double> &a, unsigned seed) {
  mt19937 gen(seed);
  uniform_real_distribution<double> dist(0.0, 1.0);
  for (double &x : a)
    x = dist(gen);
}

int main() {
  cout << "Parallel quicksort benchmark.\n";

  size_t size;
  cout << "How many random doubles? ";
  cin >> size;

  size_t cutoff;
  cout << "Sequential cutoff length for parallel recursion (for example 5000)? ";
  cin >> cutoff;
  if (cutoff == 0)
    cutoff = 1;

  int threads;
  cout << "How many OpenMP threads? ";
  cin >> threads;
  threads = max(1, threads);
  omp_set_num_threads(threads);

  unsigned seed = 4242;
  cout << "Random seed (unsigned): ";
  cin >> seed;

  vector<double> base(size);
  fill_random(base, seed);

  vector<double> seq_copy = base;
  auto t0 = chrono::high_resolution_clock::now();
  run_sequential(seq_copy);
  auto t1 = chrono::high_resolution_clock::now();

  vector<double> par_copy = base;
  auto t2 = chrono::high_resolution_clock::now();
  run_parallel(par_copy, cutoff);
  auto t3 = chrono::high_resolution_clock::now();

  cout << "\nSequential seconds: " << chrono::duration<double>(t1 - t0).count() << "\n";
  cout << "Parallel seconds: " << chrono::duration<double>(t3 - t2).count() << "\n";

  bool seq_sorted = is_sorted(seq_copy.begin(), seq_copy.end());
  bool par_sorted = is_sorted(par_copy.begin(), par_copy.end());
  bool identical = seq_copy == par_copy;

  cout << "Sequential output sorted: " << (seq_sorted ? "yes" : "no") << "\n";
  cout << "Parallel output sorted: " << (par_sorted ? "yes" : "no") << "\n";
  cout << "Outputs identical between modes: " << (identical ? "yes" : "no") << "\n";

  return 0;
}
