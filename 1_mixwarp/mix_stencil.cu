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
#include "header/stencil_header.h"

// #include "file_t/tzgemm_ori.cu"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/stencil_kernel.cu"


__global__ void mix_kernel0(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
    int stencil_grid_dim_x, int stencil_grid_dim_y, int stencil_block_dim_x, int stencil_block_dim_y,
    int stencil_iter){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, 
        MATRIX_M, MATRIX_N, MATRIX_K,
		wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < STENCIL_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_stencil0(c0, c1, A0, Anext, nx, ny, nz, 
            stencil_grid_dim_x, stencil_grid_dim_y, stencil_block_dim_x, stencil_block_dim_y,
            thread_step, stencil_iter);
    }
}


__global__ void mix_kernel1(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float c0, float c1, float *A0, float *Anext, int nx, int ny, int nz, 
    int stencil_grid_dim_x, int stencil_grid_dim_y, int stencil_block_dim_x, int stencil_block_dim_y,
    int stencil_iter){
    int thread_step = 0;
    if (threadIdx.x < wmma_block_dim_x) {
        mix_tzgemm0(a, b, c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x < wmma_block_dim_x * 2) {
        thread_step = wmma_block_dim_x * 1;
        mix_tzgemm1(a, b, c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter, thread_step);
    } else if (threadIdx.x < wmma_block_dim_x * 3) {
        thread_step = wmma_block_dim_x * 2;
        mix_stencil0(c0, c1, A0, Anext, nx, ny, nz, 
            stencil_grid_dim_x, stencil_grid_dim_y, stencil_block_dim_x, stencil_block_dim_y,
            thread_step, stencil_iter);
    } else if (threadIdx.x < wmma_block_dim_x * 4) {
        thread_step = wmma_block_dim_x * 3;
        mix_stencil1(c0, c1, A0, Anext, nx, ny, nz, 
            stencil_grid_dim_x, stencil_grid_dim_y, stencil_block_dim_x, stencil_block_dim_y,
            thread_step, stencil_iter);
    } else {
        thread_step = wmma_block_dim_x * 4;
        mix_stencil2(c0, c1, A0, Anext, nx, ny, nz, 
            stencil_grid_dim_x, stencil_grid_dim_y, stencil_block_dim_x, stencil_block_dim_y,
            thread_step, stencil_iter);
    }
}


int main(int argc, char* argv[]) {
    int stencil_blks = 3;
    int stencil_iter = 1800;
	int wmma_blks = 2;
    int wmma_iter = 1900;
    int M_INPUT = 128 * 1;
	int N_INPUT = 128 * 3136;
	int K_INPUT = 128 * 1;
	int mixwarp = 2;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        stencil_blks = atoi(argv[1]);
        stencil_iter = atoi(argv[2]);
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

    // stencil variables
    // ---------------------------------------------------------------------------------------
    float *host_stencil_ori_a0;
    float *stencil_ori_a0;
    float *stencil_ori_anext;

    float *host_stencil_ptb_a0;
    float *stencil_ptb_a0;
    float *stencil_ptb_anext;

    float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

    // nx = 128 ny = 128 nz = 32 iter = 100
    // nx = 512 ny = 512 nz = 64 iter = 100
    int nx = 128 * 4;
    int ny = 128 * 4;
    int nz = 32 * 2;

    // printf("nx: %d, ny: %d, nz: %d, iteration: %d \n", nx, ny, nz, iteration);
    host_stencil_ori_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
    cudaErrCheck(cudaMalloc((void**)&stencil_ori_a0, nx * ny * nz * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&stencil_ori_anext, nx * ny * nz * sizeof(float)));

    host_stencil_ptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
    cudaErrCheck(cudaMalloc((void**)&stencil_ptb_a0, nx * ny * nz * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&stencil_ptb_anext, nx * ny * nz * sizeof(float)));

    // curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

    curandErrCheck(curandGenerateUniform(gen, stencil_ori_a0, nx * ny * nz));
    cudaErrCheck(cudaMemcpy(stencil_ori_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(stencil_ptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(stencil_ptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));


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
    dim3 stencil_grid;
    dim3 stencil_block;
    stencil_block.x = tile_x;
    stencil_block.y = tile_y;
    stencil_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
    stencil_grid.y = (ny + tile_y - 1) / tile_y;

    printf("[ORI] Running with stencil...\n");
    printf("[ORI] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \n", stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);
    
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors(
            (ori_stencil<<<stencil_grid, stencil_block>>>(c0, c1, stencil_ori_a0, stencil_ori_anext, nx, ny, nz, stencil_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] stencil took %f ms\n\n", kernel_time);


    // PTB running
    // ---------------------------------------------------------------------------------------
    int stencil_grid_dim_x = stencil_grid.x;
    int stencil_grid_dim_y = stencil_grid.y;
    int stencil_block_dim_x = stencil_block.x;
    int stencil_block_dim_y = stencil_block.y;
    stencil_grid.x = stencil_blks == 0 ? stencil_grid_dim_x * stencil_grid_dim_y : SM_NUM * stencil_blks;
    stencil_grid.y = 1;
    stencil_block.x = stencil_block_dim_x * stencil_block_dim_y;
    stencil_block.y = 1;
    
    // printf("[PTB] Running with stencil...\n");
    // printf("[PTB] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \n", stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);

    // cudaErrCheck(cudaEventRecord(startKERNEL));
    // checkKernelErrors(
    //         (ptb_stencil<<<stencil_grid, stencil_block>>>(c0, c1, stencil_ptb_a0, stencil_ptb_anext, nx, ny, nz,
    //             stencil_grid_dim_x, stencil_grid_dim_y, stencil_block_dim_x, stencil_block_dim_y, stencil_iter)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    // printf("[PTB] stencil took %f ms\n\n", kernel_time);
    serial_time += kernel_time;


	// MIX running 
    // ----------------------------------------------------------------------------------------------------------------------
	if (mixwarp == 1) {
		dim3 mix_grid, mix_block;
        // mix_grid.x = (stencil_grid.x > wmma_grid.x) ? stencil_grid.x : wmma_grid.x;
        // mix_grid.y = 1;
        // mix_block.x = stencil_block.x + wmma_block.x;
        // mix_block.y = 1;

        mix_grid.x = SM_NUM * 2;
        mix_block.x = 128 * 2;
        printf("[PTB] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \n", 
            stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);
		printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel0 <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// stencil parameters
			c0, c1, stencil_ptb_a0, stencil_ptb_anext, nx, ny, nz,
            stencil_grid_dim_x, stencil_grid_dim_y, stencil_block_dim_x, stencil_block_dim_y, stencil_iter
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
		checkKernelErrors((ptb_stencil<<<stencil_grid, stencil_block, 0, streams[1]>>>(c0, c1, 
                stencil_ptb_a0, stencil_ptb_anext, nx, ny, nz,
                stencil_grid_dim_x, stencil_grid_dim_y, stencil_block_dim_x, stencil_block_dim_y, stencil_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \n", 
            stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
    	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);

        stencil_block.x = tile_x;
        stencil_block.y = tile_y;
        stencil_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
        stencil_grid.y = (ny + tile_y - 1) / tile_y;

        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_stencil<<<stencil_grid, stencil_block, 0, streams[1]>>>(c0, c1, 
                stencil_ptb_a0, stencil_ptb_anext, nx, ny, nz,
                stencil_iter)));
		
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
    cudaErrCheck(cudaMemcpy(host_stencil_ori_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_stencil_ptb_a0, stencil_ptb_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < nx * ny * nz; i++) {
        float v1 = host_stencil_ori_a0[i];
        float v2 = host_stencil_ptb_a0[i];
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