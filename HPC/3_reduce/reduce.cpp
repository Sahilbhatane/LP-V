/*
 * Parallel reductions (min, max, sum, average) with OpenMP — fixed small array (lab style).
 *
 * Build: g++ -O2 -fopenmp -Wall -std=c++17 -o reduce reduce.cpp
 *
 * Note: MSVC may not support min/max reductions; use GCC on Linux/WSL/MSYS2.
 */

#include <iostream>
#include <omp.h>

using namespace std;

int minval(int arr[], int n) {
  int mn = arr[0];
#pragma omp parallel for reduction(min : mn)
  for (int i = 0; i < n; i++) {
    if (arr[i] < mn)
      mn = arr[i];
  }
  return mn;
}

int maxval(int arr[], int n) {
  int mx = arr[0];
#pragma omp parallel for reduction(max : mx)
  for (int i = 0; i < n; i++) {
    if (arr[i] > mx)
      mx = arr[i];
  }
  return mx;
}

int sum_arr(int arr[], int n) {
  int s = 0;
#pragma omp parallel for reduction(+ : s)
  for (int i = 0; i < n; i++)
    s += arr[i];
  return s;
}

double average_val(int arr[], int n) { return static_cast<double>(sum_arr(arr, n)) / n; }

int main() {
  const int n = 5;
  int arr[] = {1, 2, 3, 4, 5};

  cout << "The minimum value is: " << minval(arr, n) << '\n';
  cout << "The maximum value is: " << maxval(arr, n) << '\n';
  cout << "The summation is: " << sum_arr(arr, n) << '\n';
  cout << "The average is: " << average_val(arr, n) << '\n';

  return 0;
}
