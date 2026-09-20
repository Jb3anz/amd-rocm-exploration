# HIP Programming & Porting CUDA

Notes from the AMD ROCm certification (Level 1). I wrote a HIP kernel, ported a CUDA program with hipify, and wrote a parallel reduction. Key take away: code can compile fine on AMD and still be wrong if it assumes a warp size of 32.

## Files

- `vec_add_hip.cpp` - vector add, written from scratch
- `histogram_cuda.cu` - the CUDA histogram I had to port
- `histogram_hip.cpp` - what hipify-perl gave me
- `reduction.cpp` - multi-pass sum reduction

## What I did

### Vector addition
Wrote the whole HIP flow: allocate on the GPU, copy data over, launch, copy back, check, free.

- Grid size is `(N + 255) / 256`. The `if (i < N)` check is there because the last block has extra threads that would write out of bounds.
- Made a `HIP_CHECK` macro because GPU errors are easy to miss otherwise.
- After a launch you need both `hipGetLastError()` (bad launch config) and `hipDeviceSynchronize()` (errors during the kernel).
- Got 0 errors on 1,048,576 elements.

### Porting the histogram
```bash
hipify-perl histogram_cuda.cu -o histogram_hip.cpp
hipcc -o histogram_hip histogram_hip.cpp && ./histogram_hip
```

What hipify changed:

| CUDA | HIP |
|------|-----|
| `cuda_runtime.h` | `hip/hip_runtime.h` |
| `cudaMalloc` / `cudaFree` | `hipMalloc` / `hipFree` |
| `cudaMemcpy` | `hipMemcpy` |
| `cudaMemset` | `hipMemset` |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` |

The kernel code didn't change at all (`__shared__`, `atomicAdd`, `__syncthreads`, the `<<<>>>` launch). It was mostly renaming host API calls. Output was correct: `bins[v] == v` for every bin.

### Reduction
Each thread loads one value into shared memory, then half the threads add into the other half, over and over, until one sum is left per block. One block only gives one partial sum, so I ran it in passes until one value remained. GPU and CPU both got 1,048,576.

## Warp vs wavefront

NVIDIA warps are always 32 threads. AMD wavefronts depend on the GPU: 64 on CDNA (like the MI300X), 32 on some others. So hardcoding 32 is risky, and `warpSize` (or `blockDim.x`) is the safer choice.

My kernels use `blockDim.x` and `gridDim.x` for their loops, so they don't care about warp size. I haven't written a warp-level version with `__shfl_down` yet, which is where `warpSize` would actually matter.

## Build

```bash
hipcc -o vec_add_hip vec_add_hip.cpp && ./vec_add_hip
hipify-perl histogram_cuda.cu -o histogram_hip.cpp
hipcc -o histogram_hip histogram_hip.cpp && ./histogram_hip
hipcc -o reduction reduction.cpp && ./reduction
```
