/**
 * Parallel bubble sort (odd-even transposition) and parallel merge sort using OpenMP.
 *
 * Build:
 *   g++ -O2 -fopenmp -Wall -std=c++17 -o par_sort par_sort.cpp
 *
 * Run:
 *   ./par_sort
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

using namespace std;

/**
 * Sequential bubble sort used as the correctness baseline.
 */
static void bubble_sort_seq(vector<double> &a) {
  size_t n = a.size();
  for (size_t i = 0; i < n; ++i) {
    bool swapped = false;
    for (size_t j = 0; j + 1 < n - i; ++j) {
      if (a[j] > a[j + 1]) {
        swap(a[j], a[j + 1]);
        swapped = true;
      }
    }
    if (!swapped)
      break;
  }
}

/**
 * Parallel odd-even transposition sort using OpenMP parallel for each phase.
 */
static void bubble_sort_par(vector<double> &a) {
  size_t n = a.size();
  for (size_t phase = 0; phase < n; ++phase) {
    size_t parity = phase % 2;
#pragma omp parallel for schedule(static)
    for (long long k = static_cast<long long>(parity); k < static_cast<long long>(n) - 1;
         k += 2) {
      size_t idx = static_cast<size_t>(k);
      if (a[idx] > a[idx + 1])
        swap(a[idx], a[idx + 1]);
    }
    bool sorted_flag = true;
    for (size_t i = 0; i + 1 < n; ++i) {
      if (a[i] > a[i + 1]) {
        sorted_flag = false;
        break;
      }
    }
    if (sorted_flag)
      break;
  }
}

/**
 * Merge two sorted ranges into output buffer.
 */
static void merge_ranges(const vector<double> &left, const vector<double> &right,
                         vector<double> &out) {
  size_t i = 0;
  size_t j = 0;
  out.clear();
  while (i < left.size() && j < right.size()) {
    if (left[i] <= right[j])
      out.push_back(left[i++]);
    else
      out.push_back(right[j++]);
  }
  while (i < left.size())
    out.push_back(left[i++]);
  while (j < right.size())
    out.push_back(right[j++]);
}

/**
 * Sequential merge sort wrapper.
 */
static void merge_sort_seq_impl(vector<double> &work, vector<double> &buffer, size_t left,
                                size_t right) {
  if (right - left <= 1)
    return;
  size_t mid = left + (right - left) / 2;
  merge_sort_seq_impl(work, buffer, left, mid);
  merge_sort_seq_impl(work, buffer, mid, right);

  vector<double> L(work.begin() + left, work.begin() + mid);
  vector<double> R(work.begin() + mid, work.begin() + right);
  vector<double> merged;
  merge_ranges(L, R, merged);
  copy(merged.begin(), merged.end(), work.begin() + left);
}

static void merge_sort_seq(vector<double> &data) {
  vector<double> buffer(data.size());
  merge_sort_seq_impl(data, buffer, 0, data.size());
}

/**
 * Parallel merge sort: sort chunks in parallel with OpenMP tasks, then merge.
 */
static void merge_sort_par(vector<double> &data, int threads_hint) {
  long long n = static_cast<long long>(data.size());
  if (n <= 1)
    return;

  long long chunk_count = max(1LL, static_cast<long long>(threads_hint));
  long long chunk_size = (n + chunk_count - 1) / chunk_count;

#pragma omp parallel num_threads(static_cast<int>(chunk_count))
  {
#pragma omp single
    {
      for (long long start = 0; start < n; start += chunk_size) {
        long long end = min(start + chunk_size, n);
#pragma omp task shared(data)
        {
          vector<double> slice(data.begin() + start, data.begin() + end);
          merge_sort_seq(slice);
          copy(slice.begin(), slice.end(), data.begin() + start);
        }
      }
#pragma omp taskwait
    }
  }

  // Merge sorted chunks pairwise until one array remains
  long long block = chunk_size;
  while (block < n) {
#pragma omp parallel for schedule(static)
    for (long long start = 0; start < n; start += 2 * block) {
      long long mid = min(start + block, n);
      long long end = min(start + 2 * block, n);
      if (mid >= end)
        continue;
      vector<double> L(data.begin() + start, data.begin() + mid);
      vector<double> R(data.begin() + mid, data.begin() + end);
      vector<double> merged;
      merge_ranges(L, R, merged);
      copy(merged.begin(), merged.end(), data.begin() + start);
    }
    block *= 2;
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
  cout << "Parallel sorting benchmark (bubble and merge).\n";
  size_t size;
  cout << "How many random doubles (for example 5000)? ";
  cin >> size;

  int threads;
  cout << "How many OpenMP threads should we target for parallel sorts? ";
  cin >> threads;
  threads = max(1, threads);
  omp_set_num_threads(threads);

  unsigned seed = 12345;
  cout << "Random seed (unsigned integer): ";
  cin >> seed;

  vector<double> base(size);
  fill_random(base, seed);

  vector<double> seq_bub = base;
  auto t0 = chrono::high_resolution_clock::now();
  bubble_sort_seq(seq_bub);
  auto t1 = chrono::high_resolution_clock::now();

  vector<double> par_bub = base;
  auto t2 = chrono::high_resolution_clock::now();
  bubble_sort_par(par_bub);
  auto t3 = chrono::high_resolution_clock::now();

  cout << "\nSequential bubble seconds: "
       << chrono::duration<double>(t1 - t0).count() << "\n";
  cout << "Parallel bubble seconds: "
       << chrono::duration<double>(t3 - t2).count() << "\n";

  vector<double> seq_merge = base;
  auto t4 = chrono::high_resolution_clock::now();
  merge_sort_seq(seq_merge);
  auto t5 = chrono::high_resolution_clock::now();

  vector<double> par_merge = base;
  auto t6 = chrono::high_resolution_clock::now();
  merge_sort_par(par_merge, threads);
  auto t7 = chrono::high_resolution_clock::now();

  cout << "\nSequential merge seconds: "
       << chrono::duration<double>(t5 - t4).count() << "\n";
  cout << "Parallel merge seconds: "
       << chrono::duration<double>(t7 - t6).count() << "\n";

  bool bub_ok = seq_bub == par_bub;
  bool merge_ok = seq_merge == par_merge;
  cout << "\nBubble parallel matches sequential: " << (bub_ok ? "yes" : "no") << "\n";
  cout << "Merge parallel matches sequential: " << (merge_ok ? "yes" : "no") << "\n";

  return 0;
}
