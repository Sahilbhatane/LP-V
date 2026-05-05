# LP-V practicals

This repository holds **high-performance computing (HPC)** exercises in Python and C++ with OpenMP, and **deep learning (DL)** Jupyter notebooks. The goal is hands-on practice: compare sequential vs parallel behaviour, measure timings, and train small neural networks on classic datasets. Everything is documented for **Ubuntu** with **Python 3**, **g++** and OpenMP, **Visual Studio Code**, or **Jupyter Notebook / JupyterLab**. On Windows you can run the same Python and notebooks in a virtual environment; HPC C++ programs still need a compiler with OpenMP (for example **MSYS2 MinGW** or **WSL** with `g++`).

**Repository layout**

- **`HPC/`** — Four numbered problem folders. Each contains a pair of programs with the **same base name**: `*.py` (stdlib parallelism) and `*.cpp` (OpenMP). Students answer prompts from **stdin** so inputs stay reproducible.
- **`DL/`** — Three notebooks only (no duplicate scripts). Each folder ships its own **`req.txt`** copy so you can isolate dependencies per assignment.
- Root **`req.txt`** — Aggregated Python dependencies for deep learning; subfolders may trim or duplicate this list.

---

## Requirements file (`req.txt`)

- **`req.txt`** at the project root lists the Python stack used by the DL notebooks: typically **TensorFlow**, **NumPy**, **pandas**, **scikit-learn**, and **matplotlib**. Versions are left unpinned so you can align them with your lab’s TensorFlow build (CPU vs GPU); pin versions in your own fork if you need reproducible installs.
- The same filename is **copied into each subfolder** so you can create a virtual environment **inside one practical folder** and install only what that folder needs:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install --upgrade pip
  pip install -r req.txt
  ```

- **HPC Python scripts** use only the **standard library** (`multiprocessing`, `threading`, `concurrent.futures`, etc.). Their local `req.txt` files state that **no pip packages are required** for the `.py` files; you still need Python 3 and a working terminal.

### Deep learning folders

| Location | Purpose |
|----------|---------|
| `DL/req.txt` | Shared DL dependencies for reference |
| `DL/5.Boston housing price/req.txt` | Boston housing regression notebook |
| `DL/6.IMDB/req.txt` | IMDB sentiment notebook (includes pandas/sklearn) |
| `DL/7.Fashion_MNIST_CNN/req.txt` | Fashion-MNIST CNN — lighter list (no pandas/sklearn required by the notebook itself) |

### HPC folders

Each HPC problem folder contains its own `req.txt`. For Python it documents **stdlib-only** usage. The C++ sources require **`g++`** (or compatible) with **`-fopenmp`** and a C++17-capable standard library.

---

## HPC programs (Python + C++, OpenMP in C++)

All HPC code lives under `HPC/`. Each problem has its **own subdirectory**, **interactive stdin prompts**, and **matching base filenames** for Python (`.py`) and C++ (`.cpp`). Flow is: run the program, read the printed instructions, enter parameters when asked (vertex counts, array sizes, thread counts where applicable).

| Folder | Python | C++ | Topic |
|--------|--------|-----|--------|
| `HPC/1_bfs_dfs/` | `bfs_dfs.py` | `bfs_dfs.cpp` | Parallel BFS / DFS on an undirected graph |
| `HPC/2_par_sort/` | `par_sort.py` | `par_sort.cpp` | Parallel bubble (odd-even) and merge sort + timings |
| `HPC/3_reduce/` | `reduce.py` | `reduce.cpp` | Parallel min, max, sum, average reductions |
| `HPC/4_par_qsort/` | `par_qsort.py` | `par_qsort.cpp` | Parallel quicksort-style sorting + timings |

### HPC stdin inputs — simple guide

All four programs read from the **keyboard (stdin)**. After you run the executable, type each answer **when the program asks**, usually **one number per line**. Edges use **two integers on one line** (`u` then `v`). Vertices are always **`0` … n−1**.

**Python vs C++ (quick differences)**

- **`bfs_dfs`:** Python asks for **parallel workers** at the end; **C++ does not** (OpenMP runs parallel sections inside the program).
- **`par_sort`, `reduce`, `par_qsort`:** Python lets you press **Enter** for the seed to get random data each run; **C++ always needs a numeric seed**.

Below is **one full mini-example per program** (same idea you would type by hand; using **`printf`** or PowerShell **here-strings** is optional).

---

#### 1) `bfs_dfs` — build a graph, then BFS and DFS from a start vertex

You give: how many vertices, how many edges, each edge as `u v`, where to start, and (**Python only**) how many parallel workers.

**Python** — triangle on vertices 0–1–2, start at `0`, `2` workers (7 lines):

```bash
printf "3\n3\n0 1\n1 2\n2 0\n0\n2\n" | python3 bfs_dfs.py
```

**C++** — same graph and start, no workers line (6 lines):

```bash
printf "3\n3\n0 1\n1 2\n2 0\n0\n" | ./bfs_dfs
```

---

#### 2) `par_sort` — random array size, threads/workers, seed

You give: how many random floats to sort, how many threads (Python = workers; C++ = OpenMP threads), then a seed (Python may leave blank).

**Python** — sort `500` numbers, `2` workers, seed `42`:

```bash
printf "500\n2\n42\n" | python3 par_sort.py
```

**C++** — same numbers (seed required):

```bash
printf "500\n2\n42\n" | ./par_sort
```

---

#### 3) `reduce` — random sample count, threads/workers, seed

You give: how many random doubles, parallelism count, seed. The program prints sequential vs parallel **min, max, sum, average** and times.

**Python** — `1000` samples, `2` workers, seed `99`:

```bash
printf "1000\n2\n99\n" | python3 reduce.py
```

**C++**:

```bash
printf "1000\n2\n99\n" | ./reduce
```

---

#### 4) `par_qsort` — array size, sequential cutoff, threads/workers, seed

You give: how many floats; **cutoff** (small sub-arrays use sequential quicksort inside); worker/thread count; seed. **C++:** cutoff `0` is treated as `1`.

**Python** — `800` values, cutoff `400`, `2` workers, seed `7`:

```bash
printf "800\n400\n2\n7\n" | python3 par_qsort.py
```

**C++**:

```bash
printf "800\n400\n2\n7\n" | ./par_qsort
```

---

### How to run HPC Python samples

From the problem directory (so relative paths and current working directory match any future extensions):

```bash
cd HPC/1_bfs_dfs
python3 -m venv .venv
source .venv/bin/activate
# Optional: pip install -r req.txt  (file notes there are no pip deps for the .py script)
python3 bfs_dfs.py
```

Repeat for other folders, changing the directory and script name (`par_sort.py`, `reduce.py`, `par_qsort.py`). Use the **full stdin examples** in **HPC stdin inputs** above (or paste the same numbers interactively).

### How to compile and run HPC C++ (OpenMP)

Install a compiler with OpenMP support. On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential g++
```

Example for problem 1:

```bash
cd HPC/1_bfs_dfs
g++ -O2 -fopenmp -Wall -std=c++17 -o bfs_dfs bfs_dfs.cpp
./bfs_dfs
```

Use the same pattern: replace folder name and source/output binary names (`par_sort`, `reduce`, `par_qsort`). See **HPC stdin inputs** for copy-paste **`printf`** lines (C++ `bfs_dfs` omits the workers line).

**Python vs C++ parallelism**

- Pure Python does not process `#pragma omp`. The Python versions use **`multiprocessing`**, **`threading`**, or **`concurrent.futures`** to expose parallelism at the process/thread level and to compare against sequential baselines.
- The C++ programs use **OpenMP** (`#pragma omp …`) for real shared-memory parallel loops and sections.

---

## Deep learning notebooks (`.ipynb` only)

| Notebook | Dataset | Summary |
|----------|---------|---------|
| `DL/5.Boston housing price/5_Boston_House_Price_Prediction.ipynb` | `5_boston_housing.csv` (same folder) | **Regression**: normalized numeric features, small fully-connected Keras model, train/validation split. CSV loading uses **`pathlib`** so it works when the notebook’s working directory is the folder containing the CSV. |
| `DL/6.IMDB/6_IMDB.ipynb` | `IMDB_Dataset.csv` | **Binary sentiment**: **`sklearn`** `TfidfVectorizer` for sparse TF-IDF features + dense Keras layers; text preprocessing avoids the **`re`** module per assignment constraints. |
| `DL/7.Fashion_MNIST_CNN/fashion_mnist_cnn.ipynb` | Fashion-MNIST | **Image classification**: CNN with Conv2D / MaxPooling / Dense; loads data via **`keras.datasets.fashion_mnist`**, saves **`fashion_mnist_local.npz`**, then reloads with **`numpy.load`** to demonstrate an offline bundle (the notebook contains **code cells only**—no markdown headers). |

### How to run notebooks

```bash
cd DL/6.IMDB
python3 -m venv .venv
source .venv/bin/activate
pip install -r req.txt
pip install jupyter
jupyter notebook 6_IMDB.ipynb
```

In **VS Code**, open the `.ipynb` file, choose the interpreter from your virtual environment (**Python: Select Interpreter**), then run all cells or step through them. For Fashion-MNIST, open `DL/7.Fashion_MNIST_CNN/fashion_mnist_cnn.ipynb` the same way; the first execution needs network access for Keras to download weights/data unless you already have `fashion_mnist_local.npz` beside the notebook.

**Performance note:** Training runs on **CPU** by default in these examples. For faster iteration use a machine with a recent CPU (AVX helps TensorFlow wheels) or configure GPU per TensorFlow’s official install guide.

---

## Troubleshooting

### Python multiprocessing

- **`AssertionError: daemonic processes are not allowed to have children`** — On Windows especially, **nested** `multiprocessing.Pool` usage inside pool workers is invalid. The parallel quicksort Python code avoids this: worker processes only run **sequential** sort on chunks; the parent coordinates partitioning and merging.

### OpenMP / C++

- **`g++: error: unrecognized command line option '-fopenmp'`** — Install a full GCC toolchain with OpenMP, or use a different compiler flag only if your toolchain documents an equivalent (Clang often uses `-fopenmp` with `libomp`).
- **Program runs but stays sequential** — Set **`OMP_NUM_THREADS`** in the environment (`export OMP_NUM_THREADS=4`) or use **`omp_set_num_threads`** where the source exposes it, and ensure the input size is large enough for parallel regions to dominate.

### TensorFlow / Jupyter

- **`Illegal instruction` or crash on import** — CPU may lack instructions assumed by the default TensorFlow wheel (e.g. AVX). Install a build suited to your CPU, use conda-forge, or run on newer hardware.
- **`CUDA` / GPU messages** — These notebooks assume **CPU**; GPU warnings can often be ignored unless you intentionally installed GPU TensorFlow.
- **Pandas `on_bad_lines`** — Requires a recent **pandas**; upgrade with `pip install -U pandas`. The IMDB notebook may fall back for older pandas via error handling—still prefer a current environment.

### Data files missing

- **Boston housing:** Place **`5_boston_housing.csv`** in `DL/5.Boston housing price/` and start Jupyter or VS Code with that folder as the working directory (or open the notebook from that folder).
- **IMDB:** Place **`IMDB_Dataset.csv`** next to **`6_IMDB.ipynb`**.
- **Fashion-MNIST:** First full run uses Keras to download and cache data, then writes **`fashion_mnist_local.npz`** next to the notebook for the reload path. To work fully offline, copy that `.npz` file from a machine that already ran the download step.

### Python parallel sorts / quicksort

- Very small arrays or low thread counts often show **no speedup** or **slower** parallel runs because of process startup and IPC overhead in **`multiprocessing`**. Use larger inputs and repeat runs when comparing timings.

---

## Quick reference — build commands for all C++ files

Run these **from inside each `HPC/*/` directory** so any future relative resources resolve correctly.

```bash
g++ -O2 -fopenmp -Wall -std=c++17 -o bfs_dfs bfs_dfs.cpp
g++ -O2 -fopenmp -Wall -std=c++17 -o par_sort par_sort.cpp
g++ -O2 -fopenmp -Wall -std=c++17 -o reduce reduce.cpp
g++ -O2 -fopenmp -Wall -std=c++17 -o par_qsort par_qsort.cpp
```

Then execute `./bfs_dfs`, `./par_sort`, `./reduce`, or `./par_qsort` as appropriate.
