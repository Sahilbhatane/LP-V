/**
 * Bubble + merge sort, sequential vs parallel (OpenMP). Reads array size and options from stdin.
 *
 * Build: g++ -O2 -fopenmp -Wall -std=c++17 -o par_sort par_sort.cpp
 */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

void bubbleSortSeq(vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1])
        swap(arr[j], arr[j + 1]);
    }
  }
}

void bubbleSortParallel(vector<int> &arr) {
  int n = static_cast<int>(arr.size());
  for (int i = 0; i < n; i++) {
#pragma omp parallel for
    for (int j = 0; j < n - 1; j += 2) {
      if (arr[j] > arr[j + 1])
        swap(arr[j], arr[j + 1]);
    }
#pragma omp parallel for
    for (int j = 1; j < n - 1; j += 2) {
      if (arr[j] > arr[j + 1])
        swap(arr[j], arr[j + 1]);
    }
  }
}

void merge(vector<int> &arr, int l, int m, int r) {
  vector<int> left(arr.begin() + l, arr.begin() + m + 1);
  vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

  int i = 0, j = 0, k = l;
  while (i < static_cast<int>(left.size()) && j < static_cast<int>(right.size())) {
    if (left[i] <= right[j])
      arr[k++] = left[i++];
    else
      arr[k++] = right[j++];
  }
  while (i < static_cast<int>(left.size()))
    arr[k++] = left[i++];
  while (j < static_cast<int>(right.size()))
    arr[k++] = right[j++];
}

void mergeSortSeq(vector<int> &arr, int l, int r) {
  if (l < r) {
    int m = (l + r) / 2;
    mergeSortSeq(arr, l, m);
    mergeSortSeq(arr, m + 1, r);
    merge(arr, l, m, r);
  }
}

void mergeSortParallel(vector<int> &arr, int l, int r, int depth) {
  if (l < r) {
    int m = (l + r) / 2;

    if (depth <= 0) {
      mergeSortSeq(arr, l, m);
      mergeSortSeq(arr, m + 1, r);
    } else {
#pragma omp parallel sections
      {
#pragma omp section
        mergeSortParallel(arr, l, m, depth - 1);
#pragma omp section
        mergeSortParallel(arr, m + 1, r, depth - 1);
      }
    }

    merge(arr, l, m, r);
  }
}

void generateRandom(vector<int> &arr) {
  for (int &x : arr)
    x = rand() % 100000;
}

int main() {
  int size;
  cout << "How many random integers to sort? (e.g. 10000): ";
  cin >> size;
  if (size <= 0) {
    cerr << "Size must be positive.\n";
    return 1;
  }

  int threads;
  cout << "OpenMP threads for parallel sorts (e.g. 4): ";
  cin >> threads;
  threads = max(1, threads);
  omp_set_num_threads(threads);

  unsigned seed;
  cout << "Random seed (e.g. 12345): ";
  cin >> seed;
  srand(seed);

  int merge_depth;
  cout << "Parallel merge recursion depth (e.g. 4; 0 = sequential merge halves): ";
  cin >> merge_depth;
  merge_depth = max(0, merge_depth);

  vector<int> arr(static_cast<size_t>(size)), temp;
  generateRandom(arr);

  double start, end;

  temp = arr;
  start = omp_get_wtime();
  bubbleSortSeq(temp);
  end = omp_get_wtime();
  cout << "Sequential Bubble Sort Time: " << (end - start) << " sec\n";

  temp = arr;
  start = omp_get_wtime();
  bubbleSortParallel(temp);
  end = omp_get_wtime();
  cout << "Parallel Bubble Sort Time: " << (end - start) << " sec\n";

  temp = arr;
  start = omp_get_wtime();
  mergeSortSeq(temp, 0, size - 1);
  end = omp_get_wtime();
  cout << "Sequential Merge Sort Time: " << (end - start) << " sec\n";

  temp = arr;
  start = omp_get_wtime();
  mergeSortParallel(temp, 0, size - 1, merge_depth);
  end = omp_get_wtime();
  cout << "Parallel Merge Sort Time: " << (end - start) << " sec\n";

  return 0;
}
