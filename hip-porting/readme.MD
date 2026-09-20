
# HIP Programming & Porting 🚀

This folder documents my practical work for the AMD ROCm certification. In this module, I ported from CUDA to AMD’s Heterogeneous-compute Interface for Portability (HIP) environment, working through manual kernel writing, automated code migration, and parallel reduction tuning[cite: 13].

---

## What I Built & Learned

### 1. Vector Addition from Scratch (`vec_add_hip.cpp`)
* Implemented a basic custom HIP vector-addition kernel (`__global__`) to master the core GPU execution flow: device allocation, host-to-device memory copies, kernel execution, and device synchronization[cite: 13].
* Applied ceiling division grid configuration `(N + 255) / 256` to cleanly map threads across data blocks[cite: 13].
* Created reusable error-checking macros (`HIP_CHECK`) to intercept bad launch configurations and runtime exceptions immediately[cite: 13].

### 2. Automated Code Migration (`histogram_cuda.cu` $\rightarrow$ `histogram_hip.cpp`)
* Used ROCm’s automated translation utility to convert legacy CUDA code into valid HIP source files[cite: 13]:
  ```bash
  hipify-perl histogram_cuda.cu -o histogram_hip.cpp

```

* Mapped out core API adjustments required during porting:


* **Headers:** `#include <cuda_runtime.h>` $\rightarrow$ `#include <hip/hip_runtime.h>`

* **Runtime Calls:** `cudaMalloc`, `cudaMemcpy`, `cudaFree` $\rightarrow$ `hipMalloc`, `hipMemcpy`, `hipFree`

---


### 3. Parallel Reductions & Hardware Awareness (`reduction.cpp`)

* Built a multi-pass block reduction utilizing shared memory (`__shared__`) and tree-based parallel accumulations.


* **Architecture Insight (Warps vs. Wavefronts):** Analyzed execution group differences between NVIDIA architectures (fixed **32-thread warps**) and AMD architectures (flexible **wavefronts**, often 64 threads on CDNA hardware like the MI300X). Avoided hardcoding execution group bounds to guarantee cross-platform portability.



---

## Compilation & Execution

To compile and execute the vector addition program using the AMD `hipcc` compiler:

```bash
hipcc vec_add_hip.cpp -o vec_add_hip
./vec_add_hip

```
