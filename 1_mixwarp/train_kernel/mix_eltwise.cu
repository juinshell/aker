#include <stdio.h>
#include <float.h>
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


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


__global__ void MaxForward(const int nthreads, const float* bottom_data_a,
    const float* bottom_data_b, const int blob_idx, float* top_data,
    float* mask, int iteration) {
	for (int iter_t = 0; iter_t < iteration; iter_t++) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
				i < nthreads; 
				i += blockDim.x * gridDim.x) {
			float maxval = -FLT_MAX;
			int maxidx = -1;
			if (bottom_data_a[i] > bottom_data_b[i]) {
				// only update for very first bottom_data blob (blob_idx == 0)
				if (blob_idx == 0) {
					maxval = bottom_data_a[i];
					top_data[i] = maxval;
					maxidx = blob_idx;
					mask[i] = maxidx;
				}
			} else {
				maxval = bottom_data_b[i];
				top_data[i] = maxval;
				maxidx = blob_idx + 1;
				mask[i] = maxidx;
			}
		}
	}
}


int main(int argc, char* argv[]) {
    int elt_blks = 1;
	int elt_iter = 260;
    if (argc == 3) {
        elt_blks = atoi(argv[1]);
        elt_iter = atoi(argv[2]);
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

	float *bottom0;
	float *bottom1;
	float *top;
	float *max_idx;
	
	int top_n = 16;
	int top_c = 256;
	int top_h = 56;
	int top_w = 56;
	int count = top_n * top_c * top_h * top_w;

	cudaMalloc((void**)&bottom0, count * sizeof(float));
	cudaMalloc((void**)&bottom1, count * sizeof(float));
    cudaMalloc((void**)&top, count * sizeof(float));
    cudaMalloc((void**)&max_idx, count * sizeof(float));

	// curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, bottom0, count));
    curandErrCheck(curandGenerateUniform(gen, bottom1, count));
    curandErrCheck(curandGenerateUniform(gen, top, count));
    curandErrCheck(curandGenerateUniform(gen, max_idx, count));

    dim3 elt_grid;
    dim3 elt_block;

	elt_block.x = 256;
	elt_grid.x = count / 256;
    elt_grid.x = elt_blks == 0 ? count / 256 : 68 * elt_blks;

	printf("[ORI] Running with elt...\n");
    printf("[ORI] elt_grid -- %d * %d elt_block -- %d * %d \n", 
        elt_grid.x, elt_grid.y, elt_block.x, elt_block.y);

    checkKernelErrors((MaxForward<<<elt_grid, elt_block>>> (
		count, bottom0, bottom1, 0, top, max_idx, 1000
	)));
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((MaxForward<<<elt_grid, elt_block>>> (
		count, bottom0, bottom1, 0, top, max_idx, elt_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] elt took %f ms\n\n", kernel_time);

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

    printf("[PTB] Running with tzgemm...\n");
    printf("[PTB] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);

    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(
                            wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, 1000)));

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
    checkKernelErrors((MaxForward<<<elt_grid, elt_block, 0, streams[1]>>> (
		count, bottom0, bottom1, 0, top, max_idx, elt_iter
	)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[STREAMP] mix took %f ms\n\n", kernel_time);

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

    return 0;
}