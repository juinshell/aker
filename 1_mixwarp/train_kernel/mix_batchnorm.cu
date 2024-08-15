#include <stdio.h>
#include <algorithm>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>


#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
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


#include "header/tzgemm_header.h"
#include "file_t/tzgemm_kernel.cu"


__global__ void mul_kernel(const int n, const float* a, const float* b, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                    i < n; 
                    i += blockDim.x * gridDim.x) {
        y[i] = a[i] * b[i];
    }
}


__global__ void saxpy(int n, float a, float* x, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                    i < n; 
                    i += blockDim.x * gridDim.x) {
        y[i] = a*x[i] + y[i];
    }
}


__global__ void add_scalar_kernel(const int n, const float alpha, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                        i < n; 
                        i += blockDim.x * gridDim.x) {
        y[i] += alpha;
    }
}


__global__ void sqrt_kernel(const int n, const float* a, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                            i < n; 
                            i += blockDim.x * gridDim.x) {
        y[i] = sqrt(a[i]);
    }
}


__global__ void div_kernel(const int n, const float* a,
    const float* b, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                                i < n; 
                                i += blockDim.x * gridDim.x) {
        y[i] = a[i] / b[i];
    }
}


__global__ void sgemv_kernel(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y, 
                            int iteration){
    for (int iter_t = 0; iter_t < iteration; iter_t++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                                i < m; 
                                i += blockDim.x * gridDim.x) {
            float resY = 0;
            for (int j = 0; j < n; j++) {
                resY += A[i + j * lda] * x[j];
            }
            y[i] = alpha * resY + beta * y[i];
        }
    }
}


// void mysgemv_gpu_v1(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
//     dim3 grid( CEIL_DIV(m, DIM_M), 1 );
//     dim3 threads( DIM_M, 1 );
//     sgemv_kernel_v1<<<grid, threads>>>(m, n, alpha, A, lda, x, beta, y);
// }



int main(int argc, char* argv[]) {
    int bn_blks = 1;
	int bn_iter = 260;
    if (argc == 3) {
        bn_blks = atoi(argv[1]);
        bn_iter = atoi(argv[2]);
    }

    // ---------------------------------------------------------------------------------------
        int wmma_blks = 2;
        int wmma_iter = 1;
        int M_INPUT = 16 * 8 * 32;
        int N_INPUT = 16 * 8 * 24;
        int K_INPUT = 16 * 8 * 6;

        float kernel_time;
        float serial_time = 0;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
        cudaStream_t streams[2];
        for (int i = 0; i < 2; i++) {
            cudaErrCheck(cudaStreamCreate(&streams[i]));
        }

    // tzgemm variables
    // ---------------------------------------------------------------------------------------
        int MATRIX_M = (M_INPUT < 64) ? 64 : (M_INPUT / 64) * 64;
        int MATRIX_N = (N_INPUT < 64) ? 64 : (N_INPUT / 64) * 64;
        int MATRIX_K = (K_INPUT < 64) ? 64 : (K_INPUT / 64) * 64;

        int M_TILES = MATRIX_M / WMMA_M;
        int N_TILES = MATRIX_N / WMMA_N;
        int K_TILES = MATRIX_K / WMMA_K;

        printf("M_ORI: %5d MATRIX_M: %5d (%d x %d) \n", M_INPUT, MATRIX_M, WMMA_M, M_TILES);
        printf("N_ORI: %5d MATRIX_N: %5d (%d x %d) \n", N_INPUT, MATRIX_N, WMMA_N, N_TILES);
        printf("K_ORI: %5d MATRIX_K: %5d (%d x %d) \n", K_INPUT, MATRIX_K, WMMA_K, K_TILES);

        float *ori_host_A = NULL;
        float *ori_host_B = NULL;
        // float *host_wmma_ori_c = NULL;
        // float *host_wmma_ptb_c = NULL;

        half *wmma_ori_a = NULL;
        half *wmma_ori_b = NULL;
        float *wmma_ori_c = NULL;
        float *wmma_ptb_c = NULL;

        // host_wmma_ori_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
        // host_wmma_ptb_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);

        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * MATRIX_M * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * MATRIX_N * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_a), sizeof(half) * MATRIX_M * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_b), sizeof(half) * MATRIX_N * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_c), sizeof(float) * MATRIX_M * MATRIX_N));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ptb_c), sizeof(float) * MATRIX_M * MATRIX_N));

        assert(((unsigned long long)wmma_ori_a) % 128 == 0);
        assert(((unsigned long long)wmma_ori_b) % 128 == 0);
        assert(((unsigned long long)wmma_ori_c) % 128 == 0);
        assert(((unsigned long long)wmma_ptb_c) % 128 == 0);

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(gen, ori_host_A, MATRIX_M * MATRIX_K));
        curandErrCheck(curandGenerateUniform(gen, ori_host_B, MATRIX_N * MATRIX_K));
        convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_a, ori_host_A, MATRIX_M * MATRIX_K);
        convertFp32ToFp16 <<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_b, ori_host_B, MATRIX_N * MATRIX_K);
        cudaErrCheck(cudaMemset(wmma_ori_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
        cudaErrCheck(cudaMemset(wmma_ptb_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
    // ---------------------------------------------------------------------------------------

    float *matrix_a;
    float *vector_x;
    float *vector_y;

    int matrix_h = 128*68*2*8;
    int matrix_w = 32;
    cudaMalloc((void**)&matrix_a, matrix_h * matrix_w * sizeof(float));
    cudaMalloc((void**)&vector_x, matrix_h * sizeof(float));
    cudaMalloc((void**)&vector_y, matrix_h * sizeof(float));

	// curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, matrix_a, matrix_h * matrix_w));
    curandErrCheck(curandGenerateUniform(gen, vector_x, matrix_h));
    curandErrCheck(curandGenerateUniform(gen, vector_y, matrix_h));

    dim3 bn_grid;
    dim3 bn_block;

	bn_block.x = 128;
	bn_grid.x = matrix_h / 128;
    bn_grid.x = bn_blks == 0 ? matrix_h / 128 : 68 * bn_blks;

    checkKernelErrors((sgemv_kernel<<<bn_grid, bn_block>>> (
		matrix_h, matrix_w, 0.1, matrix_a,
        matrix_h, vector_x, 0.2, vector_y, 100
	)));

	printf("[ORI] Running with bn...\n");
    printf("[ORI] bn_grid -- %d * %d bn_block -- %d * %d \n", 
        bn_grid.x, bn_grid.y, bn_block.x, bn_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((sgemv_kernel<<<bn_grid, bn_block>>> (
		matrix_h, matrix_w, 0.1, matrix_a,
        matrix_h, vector_x, 0.2, vector_y, bn_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] bn took %f ms\n\n", kernel_time);

    serial_time += kernel_time;

    // SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = wmma_blks == 0 ? wmma_grid_dim_x : SM_NUM * wmma_blks;
	wmma_block.x = THREADS_PER_BLOCK;

    // int SHMEM_SZ = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
	// cudaErrCheck(cudaFuncSetAttribute(
	// 	ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	// SHMEM_SZ = 0;

    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(
                            wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, 100)));

    printf("[PTB] Running with tzgemm...\n");
    printf("[PTB] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(
                            wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[PTB] tzgemm took %f ms\n", kernel_time);
    serial_time += kernel_time;

    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(
        wmma_ori_a, wmma_ori_b, wmma_ori_c, 
        MATRIX_M, MATRIX_N, MATRIX_K,
        wmma_grid_dim_x, wmma_block_dim_x, wmma_iter
    )));
    checkKernelErrors((sgemv_kernel<<<bn_grid, bn_block, 0, streams[1]>>> (
		matrix_h, matrix_w, 0.1, matrix_a,
        matrix_h, vector_x, 0.2, vector_y, bn_iter
	)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[STREAMP] mix took %f ms\n\n", kernel_time);

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

    return 0;
}