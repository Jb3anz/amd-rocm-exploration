# AMD ROCm & GPU Acceleration Labs

A technical log and sandbox documenting my exploration of high-performance computing (HPC) and GPU acceleration using the AMD ROCm ecosystem and the **AMD Instinct MI300X**. 

This repository focuses on moving beyond high-level abstractions to understand how compute-heavy math routines and deep learning models execute at the hardware level on enterprise accelerators.

## 🛠️ Environment Setup
* **Hardware:** AMD Instinct MI300X VF GPU
* **Software:** ROCm 7.2.4, HIP 7.2.53211
* **Framework:** PyTorch (with ROCm support)

---

## 🚀 Key Explorations & Benchmarks

### 1. Matrix Multiplications & GFLOPS (`rocBLAS`)
* Benchmarked single-precision matrix-matrix multiplication (GEMM) to evaluate raw hardware throughput.
* Reached **~65.7 TFLOPS** (roughly **43.8% FP32 GEMM efficiency**), highlighting the tuning  required.

### 2. Deep Learning Convolutions & Autotuning (`MIOpen`)
* Traced the execution pipeline (`PyTorch -> MIOpen -> HIP runtime -> GPU`) for standard convolution operations.
* Analyzed the performance delta of runtime autotuning: the initial call benchmarks various candidate algorithms (~350 ms), caches the optimal path, and drops subsequent execution times down to **~3 ms**.

### 3. Signal Processing (`rocFFT`)
* Executed Fast Fourier Transforms on GPU tensors, validating that round-trip error tolerances remained strictly within expected floating-point boundaries.

### 4. Kernel Compilation & Fusion (`torch.compile`)
* Examined PyTorch’s compiler backend integration (leveraging Triton) to achieve operator fusion, which successfully cuts down redundant kernel launch overhead and memory round-trips.

---

## 📊 Performance Summary

| Benchmark / Task | Recorded Metric | Significance |
| :--- | :--- | :--- |
| **rocBLAS GEMM** | ~65.7 TFLOPS | Raw floating-point compute capacity |
| **MIOpen Autotune (1st run)** | ~5,316 ms | Dynamic algorithm search and caching phase |
| **MIOpen Cached Run** | ~3.0 ms | Optimized execution speed using the cached kernel |
| **rocFFT Error Check** | 9.54e-07 | Numerical precision validation on transform round-trips |

## 🎯 Key Takeaways
Working through these implementations provided understanding on hardware-software cooperation, memory layout optimization, and the performance impact of kernel fusion in modern deep learning pipelines
