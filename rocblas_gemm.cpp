#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
    int M = 1024, N = 1024, K = 1024;
    float alpha = 1.0f, beta = 0.1f;

    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sA);
    float *h_B = (float*)malloc(sB);
    float *h_C = (float*)malloc(sC);

    srand(42);
    for(int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for(int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    for(int i = 0; i < K * N; i++) h_C[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, sA);
    hipMalloc(&d_B, sB);
    hipMalloc(&d_C, sC);

    hipMemcpy(d_A, h_A, sA, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, sB, hipMemcpyHostToDevice);
    hipMemcpy(d_C, h_C, sC, hipMemcpyHostToDevice);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    hipEvent_t t0, t1;
    hipEventCreate(&t0);
    hipEventCreate(&t1);

    // Warm-up run
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                  M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    hipDeviceSynchronize();

    // Benchmark phase
    hipEventRecord(t0);
    for (int i = 0; i < 20; i++) {
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                      M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    }
    hipEventRecord(t1);
    hipEventSynchronize(t1);

    float ms;
    hipEventElapsedTime(&ms, t0, t1);
    ms /= 20;

    hipMemcpy(h_C, d_C, sC, hipMemcpyDeviceToHost);

    double gflops = 2.0 * M * N * K / (ms * 1e-3) / 1e9;

    printf("=== rocBLAS SGEMM ===\n");
    printf("Size:    %dx%dx%d\n", M, N, K);
    printf("Time:    %.3f ms\n", ms);
    printf("GFLOPS:  %.1f\n", gflops);

    rocblas_destroy_handle(handle);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
