#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
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

// #include "header/tzgemm_ori.h"
// #include "file_t/tzgemm_ori.cu"
#include "header/tzgemm_header.h"
#include "file_t/tzgemm_kernel.cu"

#include "file_t/pns_rand.cu"
#include "file_t/pns_kernel.cu"
#include "header/pns_header.h"


__global__ void mix_kernel(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    int* g_s, float* g_v, int* g_m, int n, int s, int seed,
	int pns_grid_dim_x, int pns_block_dim_x, int iteration){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < PNS_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_PetrinetKernel(g_s, g_v, g_m, n, s, seed,
				pns_grid_dim_x, pns_block_dim_x, thread_step, iteration);
    }
}


int main(int argc, char* argv[]) {
    int pns_blks = 2;
	int pns_iter = 560;
	int wmma_blks = 2;
    int wmma_iter = 1900;
    int M_INPUT = 128 * 16;
	int N_INPUT = 128 * 16;
	int K_INPUT = 128 * 16;
	int mixwarp = 3;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        pns_blks = atoi(argv[1]);
        pns_iter = atoi(argv[2]);
		mixwarp = atoi(argv[3]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
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

    // tcgemm variables
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
		float *host_wmma_ori_c = NULL;
		float *host_wmma_ptb_c = NULL;

		half *wmma_ori_a = NULL;
		half *wmma_ori_b = NULL;
		float *wmma_ori_c = NULL;
		float *wmma_ptb_c = NULL;

		host_wmma_ori_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
		host_wmma_ptb_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);

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

    // pns variables
    // ---------------------------------------------------------------------------------------
		N = 1000;
		s = 1000;
		t = 200;

		N2 = N+N;
		NSQUARE2 = N*N2;

		h_vars = (float*)malloc(t*sizeof(float));
		h_maxs = (int*)malloc(t*sizeof(int));
		hp_vars = (float*)malloc(t*sizeof(float));
		hp_maxs = (int*)malloc(t*sizeof(int));

		// Allocate memory
		int unit_size = NSQUARE2 * (sizeof(int)+sizeof(char)) + sizeof(float) + sizeof(int);
		int block_num = MAX_DEVICE_MEM/unit_size;

		float *host_pns_ori_vars;
		int *host_pns_ori_maxs;
		int *pns_ori_places;
		float *pns_ori_vars;
		int *pns_ori_maxs;

		host_pns_ori_vars = h_vars;
		host_pns_ori_maxs = h_maxs;
		cudaErrCheck(cudaMalloc((void **) &pns_ori_places, (unit_size - sizeof(float) - sizeof(int)) * block_num));
		cudaErrCheck(cudaMalloc((void **) &pns_ori_vars, block_num * sizeof(float)));
		cudaErrCheck(cudaMalloc((void **) &pns_ori_maxs, block_num*sizeof(int)));

		float *host_pns_ptb_vars;
		int *host_pns_ptb_maxs;
		int *pns_ptb_places;
		float *pns_ptb_vars;
		int *pns_ptb_maxs;

		host_pns_ptb_vars = h_vars;
		host_pns_ptb_maxs = h_maxs;
		cudaErrCheck(cudaMalloc((void **) &pns_ptb_places, (unit_size - sizeof(float) - sizeof(int)) * block_num));
		cudaErrCheck(cudaMalloc((void **) &pns_ptb_vars, block_num * sizeof(float)));
		cudaErrCheck(cudaMalloc((void **) &pns_ptb_maxs, block_num*sizeof(int)));
    // ---------------------------------------------------------------------------------------


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

	int SHMEM_SZ = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
	cudaErrCheck(cudaFuncSetAttribute(
		ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	if (wmma_blks != 0) {
        SHMEM_SZ = 0;
    }

    printf("[PTB] Running with tzgemm...\n");
    printf("[PTB] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[PTB] tzgemm took %f ms\n", kernel_time);
    serial_time += kernel_time;

	// SOLO running
    // ---------------------------------------------------------------------------------------
	dim3 pns_grid;  // number of blocks
	dim3 pns_block;  // each block has 256 threads
	pns_grid.x = block_num;
	pns_block.x = BLOCK_SIZE;

	printf("[ORI] Running with pns...\n");
	printf("[ORI] pns_grid -- %d * %d * %d pns_block -- %d * %d * %d\n", 
		pns_grid.x, pns_grid.y, pns_grid.z, pns_block.x, pns_block.y, pns_block.z);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ori_pns<<< pns_grid, pns_block>>> (
		pns_ori_places, pns_ori_vars, pns_ori_maxs, N, s, 5489, pns_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[ORI] pns took %f ms\n\n", kernel_time);

	// PTB running
    // ---------------------------------------------------------------------------------------
	int pns_grid_dim_x = pns_grid.x;
	int pns_block_dim_x = pns_block.x;
    pns_grid.x = pns_blks == 0 ? pns_grid_dim_x : SM_NUM * pns_blks;
	pns_block.x = pns_block_dim_x;

	// printf("[PTB] Running with pns...\n");
	// printf("[PTB] pns_grid -- %d * %d * %d pns_block -- %d * %d * %d\n", 
	// 	pns_grid.x, pns_grid.y, pns_grid.z, pns_block.x, pns_block.y, pns_block.z);

	// cudaErrCheck(cudaEventRecord(startKERNEL));
	// checkKernelErrors((ptb_pns<<< pns_grid, pns_block>>> (
	// 	pns_ptb_places, pns_ptb_vars, pns_ptb_maxs, N, s, 5489, 
	// 	pns_grid_dim_x, pns_block_dim_x, pns_iter)));
	// cudaErrCheck(cudaEventRecord(stopKERNEL));
	// cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	// cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	// printf("[PTB] pns took %f ms\n\n", kernel_time);
    serial_time += kernel_time;


	// MIX running 
    // ----------------------------------------------------------------------------------------------------------------------
	if (mixwarp == 1) {
		dim3 mix_grid, mix_block;
        mix_grid.x = (pns_grid.x > wmma_grid.x) ? pns_grid.x : wmma_grid.x;
        mix_grid.y = 1;
        mix_block.x = pns_block.x + wmma_block.x;
        mix_block.y = 1;
		printf("[PTB] pns_grid -- %d * %d * %d pns_block -- %d * %d * %d\n", 
			pns_grid.x, pns_grid.y, pns_grid.z, pns_block.x, pns_block.y, pns_block.z);
		printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// sgemm parameters
			pns_ori_places, pns_ori_vars, pns_ori_maxs, N, s, 5489, 
		    pns_grid_dim_x, pns_block_dim_x, pns_iter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 2) {
		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ptb_pns<<< pns_grid, pns_block, 0, streams[1]>>> (
			pns_ptb_places, pns_ptb_vars, pns_ptb_maxs, N, s, 5489, 
			pns_grid_dim_x, pns_block_dim_x, pns_iter)));
			
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] pns_grid -- %d * %d * %d pns_block -- %d * %d * %d\n", 
			pns_grid.x, pns_grid.y, pns_grid.z, pns_block.x, pns_block.y, pns_block.z);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
        pns_grid.x = block_num;
	    pns_block.x = BLOCK_SIZE;

        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_pns<<< pns_grid, pns_block, 0, streams[1]>>> (
						pns_ptb_places, pns_ptb_vars, pns_ptb_maxs, N, s, 5489, 
						pns_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[STREAMO] mix took %f ms\n\n", kernel_time);
    }

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

	// Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_wmma_ori_c, wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_wmma_ptb_c, wmma_ptb_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_pns_ori_maxs, pns_ori_maxs, block_num * sizeof(int), cudaMemcpyDeviceToHost));
	cudaErrCheck(cudaMemcpy(host_pns_ori_vars, pns_ori_vars, block_num * sizeof(float), cudaMemcpyDeviceToHost));
	cudaErrCheck(cudaMemcpy(host_pns_ptb_maxs, pns_ptb_maxs, block_num * sizeof(int), cudaMemcpyDeviceToHost));
	cudaErrCheck(cudaMemcpy(host_pns_ptb_vars, pns_ptb_vars, block_num * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = host_wmma_ori_c[i];
        float v2 = host_wmma_ptb_c[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[WMMA] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[WMMA] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }
    errors = 0;
    for (int i = 0; i < block_num; i++) {
        float v1 = host_pns_ori_vars[i];
        float v2 = host_pns_ptb_vars[i];
        if (fabs(v1 - v2) > 0.001f) {
			errors++;
			if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    for (int i = 0; i < block_num; i++) {
        int v1 = host_pns_ori_maxs[i];
        int v2 = host_pns_ptb_maxs[i];
        if (v1 - v2 != 0) {
			errors++;
			if (errors < 10) printf("%d %d\n", v1, v2);
        }
		if (i < 3) printf("%d %d %d\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }


    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}