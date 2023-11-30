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
#include "header_64/img_header.h"

// #include "file_t/tzgemm_ori.cu"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/img_kernel.cu"


int main(int argc, char* argv[]) {
    int img_blks = 1;
    int img_iter = 75000;
	int wmma_blks = 2;
    int wmma_iter = 1900;
    int M_INPUT = 128 * 1;
	int N_INPUT = 128 * 3136;
	int K_INPUT = 128 * 1;
	int mixwarp = 3;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        img_blks = atoi(argv[1]);
        img_iter = atoi(argv[2]);
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
        int MATRIX_M = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
        int MATRIX_N = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
        int MATRIX_K = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;

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

    // img variables
    // ---------------------------------------------------------------------------------------
        int DATA_W = 1536;
        int DATA_H = 1024;
        int BINS = 256;

        unsigned int *host_data;
        unsigned int *img_ori_data;
        unsigned int *img_ori_result;
        unsigned int *host_img_ori_result;
        unsigned int *img_ptb_data;
        unsigned int *img_ptb_result;
        unsigned int *host_img_ptb_result;

        char inpFiles[] = "../0_mybench/file_t/img_input.iml";
        host_data = (unsigned int *)malloc(DATA_W * DATA_H * sizeof(unsigned int));
        readImage(inpFiles, host_data, DATA_W * DATA_H);

        cudaErrCheck(cudaMalloc((void**)&img_ori_data, DATA_W * DATA_H * sizeof(unsigned int)));
        cudaErrCheck(cudaMalloc((void**)&img_ori_result, BINS * sizeof(unsigned int)));
        cudaErrCheck(cudaMalloc((void**)&img_ptb_data, DATA_W * DATA_H * sizeof(unsigned int)));
        cudaErrCheck(cudaMalloc((void**)&img_ptb_result, BINS * sizeof(unsigned int)));
        cudaErrCheck(cudaMemset(img_ori_result, 0, BINS * sizeof(unsigned int)));
        cudaErrCheck(cudaMemset(img_ptb_result, 0, BINS * sizeof(unsigned int)));

        host_img_ori_result = (unsigned int *)malloc(BINS*sizeof(unsigned int));
        host_img_ptb_result = (unsigned int *)malloc(BINS*sizeof(unsigned int));

        cudaErrCheck(cudaMemcpy(img_ptb_data, host_data, DATA_W * DATA_H * sizeof(unsigned int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(img_ori_data, host_data, DATA_W * DATA_H * sizeof(unsigned int), cudaMemcpyHostToDevice));
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
    // ---------------------------------------------------------------------------------------


	// SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 img_grid;
    dim3 img_block;
    img_grid.x = NUM_BLOCKS;
    img_block.x = THREADS;

    printf("[ORI] Running with img...\n");
    printf("[ORI] img_grid -- %d * %d * %d img_block -- %d * %d * %d \n", 
        img_grid.x, img_grid.y, img_grid.z, img_block.x, img_block.y, img_block.z);
    // cudaErrCheck(cudaEventRecord(startKERNEL));
    // checkKernelErrors((ori_img<<<img_grid, img_block>>>(img_ori_result, img_ori_data, DATA_H * DATA_W, BINS, img_iter)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    // printf("[ORI] img took %f ms\n\n", kernel_time);

    int grid_dimension_x = img_grid.x;
    int block_dimension_x = img_block.x;
    img_grid.x = img_blks == 0 ? grid_dimension_x : SM_NUM * img_blks;;

    // printf("[PTB] Running with img...\n");
    // printf("[PTB] img_grid -- %d * %d * %d img_block -- %d * %d * %d \n", 
    //     img_grid.x, img_grid.y, img_grid.z, img_block.x, img_block.y, img_block.z);
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ptb_img<<<img_grid, img_block>>>(img_ori_result, img_ori_data, DATA_H * DATA_W, BINS, 
                    grid_dimension_x, block_dimension_x, img_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[PTB] img took %f ms\n\n", kernel_time);
    serial_time += kernel_time;
    // ---------------------------------------------------------------------------------------


	// MIX running 
    // ----------------------------------------------------------------------------------------------------------------------
	if (mixwarp == 2) {
		cudaErrCheck(cudaEventRecord(startKERNEL));
 
        checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
            wmma_ori_a, wmma_ori_b, wmma_ori_c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter
            )));
        checkKernelErrors((ptb_img<<<img_grid, img_block, 0, streams[1]>>>(img_ptb_result, img_ptb_data, 
                    DATA_H * DATA_W, BINS, 
                    grid_dimension_x, block_dimension_x, img_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] img_grid -- %d * %d * %d img_block -- %d * %d * %d \n", 
            img_grid.x, img_grid.y, img_grid.z, img_block.x, img_block.y, img_block.z);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
        img_grid.x = NUM_BLOCKS;
        img_block.x = THREADS;

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
            wmma_ori_a, wmma_ori_b, wmma_ori_c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter
            )));
		checkKernelErrors((ori_img<<<img_grid, img_block, 0, streams[1]>>>(img_ptb_result, img_ptb_data, 
                    DATA_H * DATA_W, BINS, img_iter)));
		
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
    cudaErrCheck(cudaMemcpy(host_img_ori_result, img_ori_result, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_img_ptb_result, img_ptb_result, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < BINS; i++) {
        unsigned int v1 = host_img_ori_result[i];
        unsigned int v2 = host_img_ptb_result[i];
        if (v1 - v2 != 0) {
            errors++;
            if (errors < 10) printf("%u %u \n", v1, v2);
        }
        if (i < 3) printf("%d %u %u\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[IMG] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    } else {
        printf("[IMG] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }


    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}