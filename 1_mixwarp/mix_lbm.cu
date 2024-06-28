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


#include "header/tzgemm_header.h"
#include "header/lbm_header.h"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/lbm_kernel.cu"


__global__ void mix_kernel0 (
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float* srcGrid, float* dstGrid,
    int lbm_grid_dim_x, int lbm_grid_dim_y, int lbm_grid_dim_z,
    int lbm_block_dim_x, int lbm_block_dim_y, int lbm_block_dim_z,
    int lbm_iter){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < LBM_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_lbm(srcGrid, dstGrid,
            lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
            lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z,
            thread_step, lbm_iter);
    }
}


__global__ void mix_kernel1 (
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float* srcGrid, float* dstGrid,
    int lbm_grid_dim_x, int lbm_grid_dim_y, int lbm_grid_dim_z,
    int lbm_block_dim_x, int lbm_block_dim_y, int lbm_block_dim_z,
    int lbm_iter){
    int thread_step = 0;
    if (threadIdx.x < wmma_block_dim_x) {
        mix_tzgemm0(a, b, c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x < wmma_block_dim_x * 2) {
        thread_step = wmma_block_dim_x;
        mix_tzgemm1(a, b, c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter, thread_step);
    } else {
        thread_step = wmma_block_dim_x * 2;
        mix_lbm(srcGrid, dstGrid,
            lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
            lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z,
            thread_step, lbm_iter);
    }
}


int main(int argc, char* argv[]) {
    int lbm_blks = 1;
    int lbm_iter = 5;
	int wmma_blks = 2; // point
    int wmma_iter = 1;
    int M_INPUT = 128 * 1;
	int N_INPUT = 128 * 3136;
	int K_INPUT = 128 * 1;
	int mixwarp = 1;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        lbm_blks = atoi(argv[1]);
        lbm_iter = atoi(argv[2]);
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
    // ---------------------------------------------------------------------------------------


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


    // lbm variables
    // ---------------------------------------------------------------------------------------
        float *lbm_ori_src;
        float *lbm_ori_dst;
        float *lbm_ptb_src;
        float *lbm_ptb_dst;
        int *lbm_ptb_src_int;
        int *lbm_ptb_dst_int;
        float *host_lbm_ori_dst;
        float *host_lbm_ptb_dst;

        const size_t size = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float) + 2 * TOTAL_MARGIN * sizeof(float);

        host_lbm_ori_dst = (float *)malloc(size);
        host_lbm_ptb_dst = (float *)malloc(size);
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_src_int, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_dst_int, size));

        // curandGenerator_t gen;
        // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(gen, lbm_ori_src, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        curandErrCheck(curandGenerateUniform(gen, lbm_ori_dst, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        cudaErrCheck(cudaMemcpy(lbm_ptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_ptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_ptb_src_int, lbm_ptb_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_ptb_dst_int, lbm_ptb_dst, size, cudaMemcpyDeviceToDevice));
        lbm_ori_src += REAL_MARGIN;
        lbm_ori_dst += REAL_MARGIN;
        lbm_ptb_src += REAL_MARGIN;
        lbm_ptb_dst += REAL_MARGIN;
        lbm_ptb_src_int += REAL_MARGIN;
        lbm_ptb_dst_int += REAL_MARGIN;
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

        int SHMEM_SZ = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * 
                (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
        cudaErrCheck(cudaFuncSetAttribute(ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
        if (wmma_blks != 0) {
            SHMEM_SZ = 0;
        }

        printf("[PTB] Running with tzgemm...\n");
        printf("[PTB] wmma_grid -- %d * %d wmma_block -- %d * %d \n", 
                wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);

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
    // ---------------------------------------------------------------------------------------


	// SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 lbm_block, lbm_grid;
        lbm_block.x = SIZE_X;
        lbm_grid.x = SIZE_Y;
        lbm_grid.y = SIZE_Z;
        lbm_block.y = lbm_block.z = lbm_grid.z = 1;
        printf("[ORI] Running with lbm...\n");
        printf("[ORI] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_lbm<<<lbm_grid, lbm_block>>>(lbm_ori_src, lbm_ori_dst, lbm_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] lbm took %f ms\n\n", kernel_time);
        serial_time += kernel_time;
    // ---------------------------------------------------------------------------------------
    

	// MIX running 
    // -----------------------------------------------------------------------------------------
	if (mixwarp == 1) {
        int lbm_block_dim_x = lbm_block.x;
        int lbm_block_dim_y = lbm_block.y;
        int lbm_block_dim_z = lbm_block.z;
        int lbm_grid_dim_x = lbm_grid.x;
        int lbm_grid_dim_y = lbm_grid.y;
        int lbm_grid_dim_z = lbm_grid.z;
        lbm_grid.x = lbm_blks == 0 ? lbm_grid_dim_x * lbm_grid_dim_y : SM_NUM * lbm_blks;
        lbm_grid.y = lbm_grid.z = 1;
        lbm_block.x = lbm_block_dim_x * lbm_block_dim_y * lbm_block_dim_z;
        lbm_block.y = lbm_block.z = 1;

		dim3 mix_grid, mix_block;
        mix_grid.x = (lbm_grid.x > wmma_grid.x) ? lbm_grid.x : wmma_grid.x;
        mix_grid.y = 1;
        mix_block.x = lbm_block.x + wmma_block.x;
        mix_block.y = 1;
        printf("[PTB] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
        lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel0 <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// sgemm parameters
			lbm_ptb_src, lbm_ptb_dst,
            lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
            lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z, lbm_iter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 2) {
        int lbm_block_dim_x = lbm_block.x;
        int lbm_block_dim_y = lbm_block.y;
        int lbm_block_dim_z = lbm_block.z;
        int lbm_grid_dim_x = lbm_grid.x;
        int lbm_grid_dim_y = lbm_grid.y;
        int lbm_grid_dim_z = lbm_grid.z;
        lbm_grid.x = lbm_blks == 0 ? lbm_grid_dim_x * lbm_grid_dim_y : SM_NUM * lbm_blks;
        lbm_grid.y = lbm_grid.z = 1;
        lbm_block.x = lbm_block_dim_x * lbm_block_dim_y * lbm_block_dim_z;
        lbm_block.y = lbm_block.z = 1;

		cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
                wmma_ori_a, wmma_ori_b, wmma_ori_c, 
                MATRIX_M, MATRIX_N, MATRIX_K,
                // alpha, beta,
                wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
        checkKernelErrors((ptb_lbm_int<<<lbm_grid, lbm_block, 0, streams[1]>>>(lbm_ptb_src_int, lbm_ptb_dst_int,
                lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
                lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z, lbm_iter)));
        
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        printf("[MIX] tgemm_grid -- %d * %d tgemm_block -- %d * %d \n", 
            wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);
		printf("[MIX] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
        lbm_block.x = SIZE_X;
        lbm_grid.x = SIZE_Y;
        lbm_grid.y = SIZE_Z;
        lbm_block.y = lbm_block.z = lbm_grid.z = 1;

        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
                wmma_ori_a, wmma_ori_b, wmma_ori_c, 
                MATRIX_M, MATRIX_N, MATRIX_K,
                // alpha, beta,
                wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_lbm<<<lbm_grid, lbm_block, 0, streams[1]>>>(lbm_ptb_src, lbm_ptb_dst, lbm_iter)));
		
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
    lbm_ori_src -= REAL_MARGIN;
    lbm_ori_dst -= REAL_MARGIN;
    lbm_ptb_src -= REAL_MARGIN;
    lbm_ptb_dst -= REAL_MARGIN;
    cudaErrCheck(cudaMemcpy(host_lbm_ori_dst, lbm_ori_dst, size, cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_lbm_ptb_dst, lbm_ptb_dst, size, cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN; i++) {
        float v1 = host_lbm_ori_dst[i];
        float v2 = host_lbm_ptb_dst[i];
        if (fabs(v1 - v2) > 0.001f) {
        errors++;
        if (errors < 10) printf("%f %f\n", v1, v2);
        }
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