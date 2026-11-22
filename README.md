# Parallel Image Filtering via Matrix Convolution

C++ benchmarking study implementing a 3×3 embossing matrix-convolution grayscale image filter across **three parallel programming paradigms**—shared memory (**OpenMP**), GPU acceleration (**CUDA**), and distributed message passing (**MPI**)—using the **FreeImage** library.

📄 **Full Technical Report:** [Report (English).pdf](./Report%20(English).pdf) | [Reporte (Español).pdf](./Reporte%20(Espa%C3%B1ol).pdf)

---

## Overview

Each implementation loads an input JPEG as an 8-bit grayscale bitmap (`foto.jpg`), applies a **3×3 embossing convolution stencil** over all interior pixels `(1..height-2, 1..width-2)`, benchmarks execution time, and exports the grayscale (`fotogris.jpg`) and filtered (`fotoconv.jpg`) outputs:

```text
 3x3 Embossing Kernel
[ -2  -1   0 ]
[ -1   1   1 ]
[  0   1   2 ]
```

### Implementations (`Code files/`)

| Model | Strategy & Key Primitives |
| :--- | :--- |
| **Sequential + OpenMP** (`convol_omp.cpp`) | Nested 2D traversal parallelized with `#pragma omp parallel num_threads(t)` and `#pragma omp for collapse(2) schedule(static)` across 2, 4, 6, and 8 threads (`std::chrono::steady_clock`). |
| **CUDA** (`convol.cu`) | Caches the 3×3 kernel in GPU constant memory (`__constant__ int d_kernel[9]` via `cudaMemcpyToSymbol`), flattens `(h-2)*(w-2)` interior pixels into a 1D thread grid, and sweeps block sizes from **128 to 1,024 threads/block** on an **NVIDIA T4 (`sm_75`)**, timed end-to-end (`cudaMalloc` + H2D + kernel + D2H) via `cudaEventElapsedTime`. |
| **MPI** (`convol_mpi.cpp`) | Root rank loads the bitmap and broadcasts dimensions and pixel data (`MPI_Bcast`); ranks process contiguous pixel slices (handling uneven remainders on the last rank) with `[0, 255]` clamping and aggregate results back to rank 0 via `MPI_Gatherv` (`counts` & `displacements`), timed with `MPI_Wtime`. |

---

## Benchmark Results

### 1. OpenMP (Shared-Memory CPU)

| Scheme | Execution Time (µs) | Speedup vs. Sequential |
| :--- | :---: | :---: |
| Sequential CPU | `183,262` | `1.00x` |
| OpenMP (2 threads) | `120,726` | `1.52x` |
| OpenMP (4 threads) | `63,831` | `2.87x` |
| OpenMP (6 threads) | `61,685` | `2.97x` |
| **OpenMP (8 threads)** | **`56,473`** | **`3.25x`** |

### 2. CUDA (NVIDIA T4 GPU, `sm_75` — End-to-End incl. Transfers)

| Block Size | Execution Time (µs) | Speedup vs. Sequential |
| :--- | :---: | :---: |
| **128 threads / block** | **`19,535.3`** | **`9.38x`** |
| 256 threads / block | `22,282.3` | `8.22x` |
| 512 threads / block | `20,176.4` | `9.08x` |
| 1,024 threads / block | `20,282.7` | `9.04x` |

### 3. MPI (Message Passing — Single-Node `mpirun`)

| Processes (`-np`) | Execution Time (µs) | Observation |
| :---: | :---: | :--- |
| 1 | `1,874,273` | High baseline serialization & buffer copy cost |
| 2 | `1,147,485` | Compute scales, but broadcast/gather dominates |
| 4 | `1,108,381` | Marginal gains over 2 processes |
| **6** | **`1,043,992`** | **Best MPI runtime (~5.7x slower than sequential)** |
| 8 | `1,073,628` | Process contention & IPC overhead reverse gains |

### Key Takeaways
- **OpenMP (`3.25x`)** scales cleanly up to 4 threads before hitting diminishing returns from memory-bandwidth saturation on a lightweight 9-MAC-per-pixel kernel.
- **CUDA (`9.38x`)** achieves the fastest end-to-end runtime at **128 threads/block**, with GPU constant-memory caching avoiding repeated global-memory kernel fetches; performance stays flat across 128–1,024 threads/block because PCIe host–device transfer time dominates kernel compute time for a single pass.
- **MPI (`1.04 s` best)** clearly demonstrates the **communication-to-computation trade-off**: broadcasting the entire bitmap via `MPI_Bcast` and reassembling via `MPI_Gatherv` costs far more than the arithmetic work of a single 3×3 stencil pass.

---

## Build & Run

**Prerequisites:** `g++` (OpenMP support), `nvcc` (CUDA Toolkit), `mpic++` / OpenMPI, and `libfreeimage-dev`. Place an input image named `foto.jpg` in the working directory.

```bash
# OpenMP
g++ -O2 convol_omp.cpp -lfreeimage -fopenmp -o convol_omp
./convol_omp

# CUDA (adjust sm_75 to match your GPU architecture)
nvcc convol.cu -O2 -std=c++17 -gencode arch=compute_75,code=sm_75 -lfreeimage -o convol_cuda
./convol_cuda

# MPI
mpic++ -O2 convol_mpi.cpp -lfreeimage -o convol_mpi
mpirun -np 6 ./convol_mpi
```

---

## Repository Structure

```text
.
├── Code files/          # C++ OpenMP, CUDA, and MPI implementations
├── Report (English).pdf # Full technical write-up, code walkthrough, and plots (English)
├── Reporte (Español).pdf# Original technical write-up (Spanish)
└── README.md
```
