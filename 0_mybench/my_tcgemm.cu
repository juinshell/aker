#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)


#include <mma.h>
using namespace nvcuda; 

#include "header/wmma_header.h"

#include "file_t/wmma_kernel.cu"

int main(int argc, char* argv[]) {
    int pers_wmma_block = 1;
    int pers_wmma_iter = 1;                     // 700
    int MATRIX_M = 4096;
    int MATRIX_N = 3072;
    int MATRIX_K = 768;

    if (argc == 6) {
        pers_wmma_block = atoi(argv[1]);
        pers_wmma_iter = atoi(argv[2]);
        MATRIX_M = atoi(argv[3]);
        MATRIX_N = atoi(argv[4]);
        MATRIX_K = atoi(argv[5]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));

    // cublas handle
    // ---------------------------------------------------------------------------------------
    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

    // tcgemm variables
    // ---------------------------------------------------------------------------------------
    float *wmma_base_a;
    float *wmma_base_b;

    half *wmma_ori_a;
    half *wmma_ori_b;
    float *wmma_ori_c;
    float *wmma_pers_c;
    float *wmma_cublas_c;
    float *host_wmma_ori_c;
    float *host_wmma_pers_c;
    float *host_wmma_cublas_c;

    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

    cudaErrCheck(cudaMalloc((void**)&wmma_base_a, MATRIX_M * MATRIX_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&wmma_base_b, MATRIX_K * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&wmma_ori_a, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&wmma_ori_b, MATRIX_K * MATRIX_N * sizeof(half)));

    cudaErrCheck(cudaMalloc((void**)&wmma_cublas_c, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&wmma_pers_c, MATRIX_M * MATRIX_N * sizeof(float)));

    host_wmma_cublas_c = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    host_wmma_ori_c = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    host_wmma_pers_c = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    curandErrCheck(curandGenerateUniform(gen, wmma_base_a, MATRIX_M * MATRIX_K));
    curandErrCheck(curandGenerateUniform(gen, wmma_base_b, MATRIX_K * MATRIX_N));
    // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
    convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_a, wmma_base_a, MATRIX_M * MATRIX_K);
    convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (wmma_ori_b, wmma_base_b, MATRIX_K * MATRIX_N);
    curandErrCheck(curandDestroyGenerator(gen));

    cudaErrCheck(cudaMemset(wmma_cublas_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
	cudaErrCheck(cudaMemset(wmma_ori_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
	cudaErrCheck(cudaMemset(wmma_pers_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));

    // cudaErrCheck(cudaMemcpy(wmma_cublas_c, wmma_base_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
    // cudaErrCheck(cudaMemcpy(wmma_ori_c, wmma_base_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
    // cudaErrCheck(cudaMemcpy(wmma_pers_c, wmma_base_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

    // SOLO running
    // ---------------------------------------------------------------------------------------
    printf("M = %d, N = %d, K = %d\n\n", MATRIX_M, MATRIX_N, MATRIX_K);
    // First: using WMMA
    dim3 wmma_grid;
    dim3 wmma_block;
    // wmma_block.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    wmma_block.x = 128;
    wmma_block.y = 4;
    wmma_grid.x = (MATRIX_M + (WMMA_M * wmma_block.x / 32 - 1)) / (WMMA_M * wmma_block.x / 32);
    wmma_grid.y = (MATRIX_N + WMMA_N * wmma_block.y - 1) / (WMMA_N * wmma_block.y);

    printf("[ORI] Running with wmma...\n");
    printf("[ORI] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ori_wmma <<< wmma_grid, wmma_block >>> (wmma_ori_a, wmma_ori_b, wmma_ori_c, 
                                    MATRIX_M, MATRIX_N, MATRIX_K, 
                                    pers_wmma_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] wmma took %f ms\n\n", kernel_time);

    // PTB running
    // ---------------------------------------------------------------------------------------
    wmma_block.x = 128;
    wmma_block.y = 1;
    wmma_grid.x = (MATRIX_M + (WMMA_M * wmma_block.x / 32 - 1)) / (WMMA_M * wmma_block.x / 32);
    wmma_grid.y = (MATRIX_N + WMMA_N * wmma_block.y - 1) / (WMMA_N * wmma_block.y);
    printf("[ORI] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);
    int wmma_grid_dim_x = wmma_grid.x;
    int wmma_grid_dim_y = wmma_grid.y;
    int wmma_block_dim_x = wmma_block.x;
    int wmma_block_dim_y = wmma_block.y;

    // wmma_grid.x = wmma_grid_dim_x * wmma_grid_dim_y;
    wmma_grid.x = pers_wmma_block == 0 ? wmma_grid_dim_x * wmma_grid_dim_y : 68 * pers_wmma_block;
    wmma_grid.y = 1;
    wmma_block.x = wmma_block_dim_x * wmma_block_dim_y;
    wmma_block.y = 1;

    printf("[PERS] Running with wmma...\n");
    printf("[PERS] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((pers_wmma <<< wmma_grid, wmma_block >>> (wmma_ori_a, wmma_ori_b, wmma_pers_c,
                    MATRIX_M, MATRIX_N, MATRIX_K,
                    wmma_grid_dim_x, wmma_grid_dim_y, wmma_block_dim_x, wmma_block_dim_y,
                    pers_wmma_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[PERS] wmma took %f ms\n\n", kernel_time);

    // CUBLAS running
    // ---------------------------------------------------------------------------------------
    printf("Running with cuBLAS...\n");
    cudaErrCheck(cudaEventRecord(startKERNEL));
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_M, MATRIX_N, MATRIX_K, 
                &alpha,
                wmma_ori_a, CUDA_R_16F, MATRIX_M,
                wmma_ori_b, CUDA_R_16F, MATRIX_K,
                &beta, 
                wmma_cublas_c, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("cublas took %f ms\n", kernel_time);

    // Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_wmma_ori_c, wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_wmma_pers_c, wmma_pers_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_wmma_cublas_c, wmma_cublas_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

    // 0.01% relative tolerance. 1e-5 absolute tolerance.
    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = host_wmma_ori_c[i];
        float v2 = host_wmma_cublas_c[i];
        if (fabs(v1 - v2) > 0.01f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
        if (i < 5) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
    }
    else {
        printf("Results verified: cublas and WMMA agree.\n");
    }
    errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = host_wmma_ori_c[i];
        float v2 = host_wmma_pers_c[i];
        if (fabs(v1 - v2) > 0.01f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
    }
    if (errors > 0) {
        printf("WMMA does not agree with PERS! %d errors!\n", errors);
    }
    else {
        printf("Results verified: WMMA and PERS agree.\n");
    }

    cudaErrCheck(cudaEventDestroy(startKERNEL));             
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaFree(wmma_base_a));
    cudaErrCheck(cudaFree(wmma_base_b));
    cudaErrCheck(cudaFree(wmma_ori_a));
    cudaErrCheck(cudaFree(wmma_ori_b));

    cudaErrCheck(cudaFree(wmma_cublas_c));
    cudaErrCheck(cudaFree(wmma_ori_c));
    cudaErrCheck(cudaFree(wmma_pers_c));

    free(host_wmma_cublas_c);
    free(host_wmma_ori_c);
    free(host_wmma_pers_c);

    cudaErrCheck(cudaDeviceReset());
    return 0;
}