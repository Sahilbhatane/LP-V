# LP-V practicals

This repository holds **high-performance computing (HPC)** exercises in Python and C++ with OpenMP, and **deep learning (DL)** Jupyter notebooks. The goal is hands-on practice: compare sequential vs parallel behaviour, measure timings, and train small neural networks on classic datasets. Everything is documented for **Ubuntu** with **Python 3**, **g++** and OpenMP, **Visual Studio Code**, or **Jupyter Notebook / JupyterLab**. On Windows you can run the same Python and notebooks in a virtual environment; HPC C++ programs still need a compiler with OpenMP (for example **MSYS2 MinGW** or **WSL** with `g++`).

**Repository layout**

- **`HPC/`** — Four numbered problem folders. Each has matching **`*.py`** and **`*.cpp`** names. Only **`bfs_dfs`** reads interactive stdin; the others use **constants in source** (fixed array size or fixed data).
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

Layout matches common **SPPU-style** labs: short programs, predictable OpenMP patterns, and minimal ceremony.

| Folder | What it does |
|--------|----------------|
| `HPC/1_bfs_dfs/` | **C++:** parallel BFS + parallel DFS (`Graph` class, OpenMP). **Python:** same prompts, **sequential** BFS/DFS (easy to verify against lecture notes). |
| `HPC/2_par_sort/` | Sequential vs parallel **bubble** and **merge** sort timings. **`SIZE`** is set in each source file (C++ `10000`; Python `256` so threaded odd-even finishes quickly). **No stdin.** |
| `HPC/3_reduce/` | **Min / max / sum / average** on fixed array `{1,2,3,4,5}`. **No stdin.** |
| `HPC/4_par_qsort/` | Sequential vs parallel **quicksort** on **`N = 5000`** random integers. **No stdin.** |

### `bfs_dfs` — stdin (same questions in Python and C++)

1. Number of vertices `V`  
2. Number of edges `E`  
3. `E` lines: `u v` (must satisfy `0 ≤ u,v < V`)  
4. Starting vertex  

**Full example** (triangle on `0,1,2`, start `0`):

```bash
printf "3\n3\n0 1\n1 2\n2 0\n0\n" | python3 bfs_dfs.py
printf "3\n3\n0 1\n1 2\n2 0\n0\n" | ./bfs_dfs
```

### Other HPC programs — just run them

No typing required:

```bash
cd HPC/2_par_sort && python3 par_sort.py
cd HPC/3_reduce   && python3 reduce.py
cd HPC/4_par_qsort && python3 par_qsort.py
```

Change **`SIZE`**, **`N`**, or the **data array** in the source when you want larger experiments.

---

### How to run HPC Python samples

```bash
cd HPC/1_bfs_dfs
python3 bfs_dfs.py
```

Use `python3 par_sort.py`, `python3 reduce.py`, `python3 par_qsort.py` from each folder (no arguments).

### How to compile and run HPC C++ (OpenMP)

Install a compiler with OpenMP support. On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential g++
```

Example:

```bash
cd HPC/1_bfs_dfs
g++ -O2 -fopenmp -Wall -std=c++17 -o bfs_dfs bfs_dfs.cpp
./bfs_dfs
```

Repeat for `par_sort`, `reduce`, `par_qsort` (same flags; output binary name matches the `.cpp` basename).

**Python vs C++**

- Python uses **`multiprocessing`** / **`ThreadPoolExecutor`** where noted; it does not compile `#pragma omp`.
- C++ uses **OpenMP** (`-fopenmp`). **`reduce.cpp`** uses OpenMP reductions on `int` — prefer **GCC** on Linux/WSL/MSYS2 (MSVC OpenMP reductions are limited).

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
- **IMDB:** The large dataset **`IMDB_Dataset.csv`** is stored with **Git LFS** in this repository. Install [Git LFS](https://git-lfs.com/) (`git lfs install`) and clone or pull as usual so the real CSV downloads (not only the tiny pointer file). Alternatively place your own copy next to **`6_IMDB.ipynb`**.
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
