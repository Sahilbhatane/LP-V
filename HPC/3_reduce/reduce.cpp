/**
 * Parallel reductions for minimum, maximum, sum, and average using OpenMP.
 *
 * Build:
 *   g++ -O2 -fopenmp -Wall -std=c++17 -o reduce reduce.cpp
 *
 * Run:
 *   ./reduce
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <vector>

using namespace std;

/**
 * Sequential baseline so students can compare timings with parallel reductions.
 */
static void sequential_stats(const vector<double> &data, double &mn, double &mx,
                             double &sum_out, double &avg_out) {
  mn = data[0];
  mx = data[0];
  double sum = data[0];
  for (size_t i = 1; i < data.size(); ++i) {
    double x = data[i];
    if (x < mn)
      mn = x;
    if (x > mx)
      mx = x;
    sum += x;
  }
  sum_out = sum;
  avg_out = sum / static_cast<double>(data.size());
}

/**
 * Parallel reductions using OpenMP clauses on one parallel region.
 */
static void parallel_stats(const vector<double> &data, double &mn, double &mx,
                           double &sum_out, double &avg_out) {
  double local_min = numeric_limits<double>::infinity();
  double local_max = -numeric_limits<double>::infinity();
  double total = 0.0;

#pragma omp parallel for reduction(min : local_min) reduction(max : local_max) reduction(+ : total)
  for (long long i = 0; i < static_cast<long long>(data.size()); ++i) {
    double x = data[(size_t)i];
    local_min = min(local_min, x);
    local_max = max(local_max, x);
    total += x;
  }

  mn = local_min;
  mx = local_max;
  sum_out = total;
  avg_out = total / static_cast<double>(data.size());
}

int main() {
  cout << "Parallel reduction demo (min, max, sum, average).\n";

  size_t count;
  cout << "How many random doubles? ";
  cin >> count;

  if (count == 0) {
    cerr << "Count must be positive.\n";
    return 1;
  }

  int threads;
  cout << "How many OpenMP threads? ";
  cin >> threads;
  threads = max(1, threads);
  omp_set_num_threads(threads);

  unsigned seed = 999;
  cout << "Random seed (unsigned): ";
  cin >> seed;

  vector<double> data(count);
  mt19937 gen(seed);
  uniform_real_distribution<double> dist(-1000.0, 1000.0);
  for (double &x : data)
    x = dist(gen);

  double mn_s, mx_s, sum_s, avg_s;
  auto t0 = chrono::high_resolution_clock::now();
  sequential_stats(data, mn_s, mx_s, sum_s, avg_s);
  auto t1 = chrono::high_resolution_clock::now();

  double mn_p, mx_p, sum_p, avg_p;
  auto t2 = chrono::high_resolution_clock::now();
  parallel_stats(data, mn_p, mx_p, sum_p, avg_p);
  auto t3 = chrono::high_resolution_clock::now();

  cout << "\nSequential seconds: " << chrono::duration<double>(t1 - t0).count() << "\n";
  cout << "Parallel seconds: " << chrono::duration<double>(t3 - t2).count() << "\n";

  cout << "\nSequential min/max/sum/avg: " << mn_s << " / " << mx_s << " / " << sum_s
       << " / " << avg_s << "\n";
  cout << "Parallel min/max/sum/avg: " << mn_p << " / " << mx_p << " / " << sum_p << " / "
       << avg_p << "\n";

  bool ok = fabs(mn_s - mn_p) < 1e-9 && fabs(mx_s - mx_p) < 1e-9 && fabs(sum_s - sum_p) < 1e-6 &&
            fabs(avg_s - avg_p) < 1e-9;
  cout << "Results match within tolerance: " << (ok ? "yes" : "no") << "\n";

  return 0;
}
