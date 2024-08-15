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
#include "header/cp_header.h"

#include "file_t/tzgemm_kernel.cu"
#include "file_t/cp_kernel.cu"


__global__ void mix_kernel0(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    int numatoms, float gridspacing, float * energygrid, 
	int cp_grid_dim_x, int cp_grid_dim_y, int cp_block_dim_x, int cp_block_dim_y,
	int cp_iter){
    if (threadIdx.x < 128 * 1) {
        mix_tzgemm0(a, b, c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, 128, wmma_iter);
    } else if (threadIdx.x < 128 * 2) {
        mix_cp0(numatoms, gridspacing, energygrid, 
			cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, 128,
			cp_iter);
    } else {
        mix_cp1(numatoms, gridspacing, energygrid, 
			cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, 128 * 2,
			cp_iter);
    }
}


__global__ void mix_kernel1(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    int numatoms, float gridspacing, float * energygrid, 
	int cp_grid_dim_x, int cp_grid_dim_y, int cp_block_dim_x, int cp_block_dim_y,
	int cp_iter){
    if (threadIdx.x < 128 * 1) {
        mix_tzgemm0(a, b, c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, 128, wmma_iter);
    } else {
        mix_cp0(numatoms, gridspacing, energygrid, 
			cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, 128,
			cp_iter);
    }
}


int main(int argc, char* argv[]) {
    int cp_blks = 4;
	int cp_iter = 1;
	int wmma_blks = 2;
    int wmma_iter = 1;
    int M_INPUT = 50176;
	int N_INPUT = 512;
	int K_INPUT = 128 * 10;
	int mixwarp = 1;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        cp_blks = atoi(argv[1]);
        cp_iter = atoi(argv[2]);
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


    // cp variables
    // ---------------------------------------------------------------------------------------
		float *atoms = NULL;
		int atomcount = ATOMCOUNT;
		const float gridspacing = 0.1;					// number of atoms to simulate
		dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
		initatoms(&atoms, atomcount, volsize, gridspacing);

		// allocate and initialize the GPU output array
		int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

		float *ori_output;	
		float *ptb_output;
		cudaErrCheck(cudaMalloc((void**)&ori_output, volmemsz));
		cudaErrCheck(cudaMemset(ori_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
		cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
		float *host_ori_energy = (float *) malloc(volmemsz);
		float *host_ptb_energy = (float *) malloc(volmemsz);
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
		cudaErrCheck(cudaFuncSetAttribute(
			ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
		if (wmma_blks != 0) {
			SHMEM_SZ = 0;
		}
		printf("wmma_grid_dim_x: %d wmma_block_dim_x: %d\n", wmma_grid_dim_x, wmma_block_dim_x);
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
    // ---------------------------------------------------------------------------------------


	// SOLO running
    // ---------------------------------------------------------------------------------------
		dim3 cp_grid, cp_block;
		cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
		cp_block.y = BLOCKSIZEY;
		cp_block.z = 1;
		cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
		cp_grid.y = volsize.y / cp_block.y; 
		cp_grid.z = volsize.z / cp_block.z; 
		printf("[ORI] Running with cp...\n");
		printf("[ORI] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		int atomstart = 1;
		int runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_cp <<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ori_output, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] cp took %f ms\n\n", kernel_time);
		serial_time += kernel_time;
    // ---------------------------------------------------------------------------------------
	

	// MIX running 
    // ---------------------------------------------------------------------------------------
	if (mixwarp == 1) {
		int cp_grid_dim_x = cp_grid.x;
		int cp_grid_dim_y = cp_grid.y;
		int cp_block_dim_x = cp_block.x;
		int cp_block_dim_y = cp_block.y;
		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		dim3 mix_grid, mix_block;
		// mix_grid.x = SM_NUM * 2;
		// mix_block.x = cp_block.x * cp_block.y * 2 + wmma_block.x * 1;
		mix_grid.x = SM_NUM * 2;
		mix_block.x = cp_block.x * cp_block.y + wmma_block.x * 1;
		printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel0 <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// cp parameters
			runatoms, 0.1, ptb_output, 
			cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, cp_iter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 2) {
		int cp_grid_dim_x = cp_grid.x;
		int cp_grid_dim_y = cp_grid.y;
		int cp_block_dim_x = cp_block.x;
		int cp_block_dim_y = cp_block.y;
		cp_grid.x = cp_blks == 0 ? cp_grid_dim_x * cp_grid_dim_y : SM_NUM * cp_blks;
		cp_grid.y = 1;
		cp_block.x = cp_block_dim_x * cp_block_dim_y;
		cp_block.y = 1;
		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
				wmma_ori_a, wmma_ori_b, wmma_ori_c,
				MATRIX_M, MATRIX_N, MATRIX_K,
				wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ptb_cp <<<cp_grid, cp_block, 0, streams[1]>>>(
				runatoms, 0.1, ptb_output,
				cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
				cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
        cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
        cp_block.y = BLOCKSIZEY;
        cp_block.z = 1;
        cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
        cp_grid.y = volsize.y / cp_block.y;
        cp_grid.z = volsize.z / cp_block.z;

        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(
				wmma_ori_a, wmma_ori_b, wmma_ori_c, 
				MATRIX_M, MATRIX_N, MATRIX_K,
				wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_cp <<<cp_grid, cp_block, 0, streams[1]>>>(
				runatoms, 0.1, ptb_output, cp_iter)));
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
    cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
	cudaMemcpy(host_ptb_energy, ptb_output, volmemsz,  cudaMemcpyDeviceToHost);

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
	for (int i = 0; i < volsize.x * volsize.y * volsize.z; i++) {
		float v1 = host_ori_energy[i];
		float v2 = host_ptb_energy[i];
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