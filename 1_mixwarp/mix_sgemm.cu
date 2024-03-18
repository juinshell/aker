#include <stdio.h>
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
#include "header/tzgemm_header.h"
#include "header/sgemm_header.h"

// #include "file_t/tzgemm_ori.cu"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/sgemm_kernel.cu"


__global__ void mix_kernel(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
    int sgemm_grid_dim_x, int sgemm_grid_dim_y, int sgemm_block_dim_x, int sgemm_block_dim_y,
    int sgemm_iter){
    if (threadIdx.x < wmma_block_dim_x && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K, wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x && blockIdx.x < SGEMM_GRID_DIM) {
        int thread_step = wmma_block_dim_x;
        mix_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, 
                sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter, thread_step);
    }
}


__global__ void mix_kernel1(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
    int sgemm_grid_dim_x, int sgemm_grid_dim_y, int sgemm_block_dim_x, int sgemm_block_dim_y,
    int sgemm_iter){
    int thread_step = 0;
    if (threadIdx.x < wmma_block_dim_x) {
        mix_tzgemm0(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K, wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x < wmma_block_dim_x * 2) {
        thread_step = wmma_block_dim_x;
        mix_tzgemm1(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K, wmma_grid_dim_x, wmma_block_dim_x, wmma_iter, thread_step);
    } else {
        thread_step = wmma_block_dim_x * 2;
        mix_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, 
                sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter, thread_step);
    } 
}

__global__ void mix_kernel2(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float *A, float *B, float *C, int NORMAL_M, int NORMAL_N, int NORMAL_K, 
    int sgemm_grid_dim_x, int sgemm_grid_dim_y, int sgemm_block_dim_x, int sgemm_block_dim_y,
    int sgemm_iter){
    int thread_step = 0;
    if (threadIdx.x < wmma_block_dim_x) {
        mix_tzgemm0(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K, wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x < wmma_block_dim_x * 2) {
        thread_step = wmma_block_dim_x;
        mix_sgemm(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, 
                sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter, thread_step);
    } else if (threadIdx.x < wmma_block_dim_x * 3) {
        thread_step = wmma_block_dim_x * 2;
        mix_sgemm1(A, B, C, NORMAL_M, NORMAL_N, NORMAL_K, 
                sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter, thread_step);
    }
}


int main(int argc, char* argv[]) {
    int sgemm_blks = 2;
    int sgemm_iter = 1;
	int wmma_blks = 2;
    int wmma_iter = 8;
    int M_INPUT = 128 * 16;
	int N_INPUT = 128 * 16;
	int K_INPUT = 128 * 16;
	int mixwarp = 1;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        sgemm_blks = atoi(argv[1]);
        sgemm_iter = atoi(argv[2]);
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


    // sgemm variables
    // ---------------------------------------------------------------------------------------
    float *sgemm_ori_a;
    float *sgemm_ori_b;
    float *sgemm_ori_c;
    float *sgemm_ptb_a;
    float *sgemm_ptb_b;
    float *sgemm_ptb_c;
    float *host_sgemm_ori_c;
    float *host_sgemm_ptb_c;

    int NORMAL_M = 4096;
    int NORMAL_N = 4096;
    int NORMAL_K = 256;

    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float)));

    host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
    host_sgemm_ptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));

    // curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
    curandErrCheck(curandGenerateUniform(gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
    cudaErrCheck(cudaMemcpy(sgemm_ptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(sgemm_ptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
    curandErrCheck(curandDestroyGenerator(gen));


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
	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
                            wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[PTB] tzgemm took %f ms\n", kernel_time);
    serial_time += kernel_time;

	// SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 sgemm_grid;
    dim3 sgemm_block;
    sgemm_block.x = TILE_N;
    sgemm_block.y = TILE_TB_HEIGHT;
    sgemm_grid.x = NORMAL_M/TILE_M;
    sgemm_grid.y = NORMAL_N/TILE_N;
    printf("[ORI] Running with sgemm...\n");
    printf("[ORI] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
        sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block >>> (sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
                            NORMAL_M, NORMAL_N, NORMAL_K, sgemm_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] sgemm took %f ms\n\n", kernel_time);

    // PTB running
    // ---------------------------------------------------------------------------------------
    int sgemm_grid_dim_x = sgemm_grid.x;
    int sgemm_grid_dim_y = sgemm_grid.y;
    int sgemm_block_dim_x = sgemm_block.x;
    int sgemm_block_dim_y = sgemm_block.y;
    sgemm_grid.x = sgemm_blks == 0 ? sgemm_grid_dim_x * sgemm_grid_dim_y : SM_NUM * sgemm_blks;
    sgemm_grid.y = 1;
    sgemm_block.x = sgemm_block_dim_x * sgemm_block_dim_y;
    sgemm_block.y = 1;
    // printf("[PTB] Running with sgemm...\n");
    // printf("[PTB] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
    //     sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

    // cudaErrCheck(cudaEventRecord(startKERNEL));
    // checkKernelErrors((ptb_sgemm <<< sgemm_grid, sgemm_block >>> (
    //                 sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
	// 				NORMAL_M, NORMAL_N, NORMAL_K,
    //                 sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    // printf("[PTB] sgemm took %f ms\n\n", kernel_time);
    serial_time += kernel_time;

	// MIX running 
    // ----------------------------------------------------------------------------------------------------------------------
	if (mixwarp == 1) {
		dim3 mix_grid;
		dim3 mix_block;
		mix_grid.x = (sgemm_grid.x > wmma_grid.x) ? sgemm_grid.x : wmma_grid.x;
        mix_grid.x = SM_NUM * 1;
		mix_grid.y = 1;
		mix_block.x = sgemm_block.x * 2 + wmma_block.x;
		mix_block.y = 1;
        printf("[PTB] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);
		printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel2 <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, 
            wmma_iter,
			// sgemm parameters
			sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
            NORMAL_M, NORMAL_N, NORMAL_K,
            sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 2) {
		cudaErrCheck(cudaEventRecord(startKERNEL));

        checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
            wmma_ori_a, wmma_ori_b, wmma_ori_c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter
        )));
        checkKernelErrors((ptb_sgemm <<< sgemm_grid, sgemm_block, 0, streams[1] >>> (
            sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
            NORMAL_M, NORMAL_N, NORMAL_K,
            sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter)));
        
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
        sgemm_block.x = TILE_N;
        sgemm_block.y = TILE_TB_HEIGHT;
        sgemm_grid.x = NORMAL_M/TILE_M;
        sgemm_grid.y = NORMAL_N/TILE_N;

        cudaErrCheck(cudaEventRecord(startKERNEL));

        checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
            wmma_ori_a, wmma_ori_b, wmma_ori_c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
        checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block, 0, streams[1] >>> (
            sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
            NORMAL_M, NORMAL_N, NORMAL_K,
            sgemm_iter)));
		
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
    cudaErrCheck(cudaMemcpy(host_sgemm_ori_c, sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_sgemm_ptb_c, sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < NORMAL_M * NORMAL_N; i++) {
        float v1 = host_sgemm_ori_c[i];
        float v2 = host_sgemm_ptb_c[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[SGEMM] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[SGEMM] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}