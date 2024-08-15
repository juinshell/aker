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
#include "header/tpacf_header.h"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/tpacf_kernel.cu"


__global__ void mix_kernel0(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    hist_t* histograms, float* all_x_data, float* all_y_data, 
    float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
    int tpacf_grid_dim_x, int tpacf_grid_dim_y, int tpacf_block_dim_x, int tpacf_block_dim_y, 
    int iteration){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, 
        MATRIX_M, MATRIX_N, MATRIX_K,
		wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < TPACF_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_tpacf(histograms, all_x_data, all_y_data, 
            all_z_data, NUM_SETS, NUM_ELEMENTS,
            tpacf_grid_dim_x, tpacf_grid_dim_y, tpacf_block_dim_x, tpacf_block_dim_y, 
            thread_step, iteration);
    }
}


__global__ void mix_kernel1(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    hist_t* histograms, float* all_x_data, float* all_y_data, 
    float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
    int tpacf_grid_dim_x, int tpacf_grid_dim_y, int tpacf_block_dim_x, int tpacf_block_dim_y, 
    int iteration){
    int thread_step = 0;
    if (threadIdx.x < wmma_block_dim_x * 1) {
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
        mix_tpacf(histograms, all_x_data, all_y_data, 
            all_z_data, NUM_SETS, NUM_ELEMENTS,
            tpacf_grid_dim_x, tpacf_grid_dim_y, tpacf_block_dim_x, tpacf_block_dim_y, 
            thread_step, iteration);
    }
}


int main(int argc, char* argv[]) {
    int tpacf_blks = 3;
    int tpacf_iter = 1;
	int wmma_blks = 2;
    int wmma_iter = 300;
    int M_INPUT = 128 * 1;
	int N_INPUT = 128 * 3136;
	int K_INPUT = 128 * 1;
	int mixwarp = 1;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        tpacf_blks = atoi(argv[1]);
        tpacf_iter = atoi(argv[2]);
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


    // tpacf variables
    // ---------------------------------------------------------------------------------------
        // 10391
        // 97178
        // NUM_ELEMENTS = 97178;
        int NUM_ELEMENTS = 4096;
        int NUM_SETS = 100;
        int num_elements = NUM_ELEMENTS; 
        unsigned f_mem_size = (1 + NUM_SETS) * num_elements * sizeof(float);
        float *binb = (float *)malloc((NUM_BINS+1)*sizeof(float));
        for (int k = 0; k < NUM_BINS+1; k++){
            binb[k] = cos(pow(10.0, (log10(min_arcmin) + k*1.0/bins_per_dec)) / 60.0*D2R);
        }

        hist_t *tpacf_ori_hists;
        float *tpacf_ori_x;
        float *tpacf_ori_y;
        float *tpacf_ori_z;
        hist_t *tpacf_ptb_hists;
        float *tpacf_ptb_x;
        float *tpacf_ptb_y;
        float *tpacf_ptb_z;
        hist_t *host_tpacf_ori_hists;
        hist_t *host_tpacf_ptb_hists;

    
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMemset(tpacf_ori_hists, 100, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_x, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_y, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_z, f_mem_size));

        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMemset(tpacf_ptb_hists, 100, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_x, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_y, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_z, f_mem_size));

        host_tpacf_ori_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t));
        host_tpacf_ptb_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t));

        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(gen, tpacf_ori_x, (1 + NUM_SETS) * num_elements));
        curandErrCheck(curandGenerateUniform(gen, tpacf_ori_y, (1 + NUM_SETS) * num_elements));
        curandErrCheck(curandGenerateUniform(gen, tpacf_ori_z, (1 + NUM_SETS) * num_elements));

        cudaErrCheck(cudaMemcpy(tpacf_ptb_x, tpacf_ori_x, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_ptb_y, tpacf_ori_y, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_ptb_z, tpacf_ori_z, f_mem_size, cudaMemcpyDeviceToDevice));
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
        dim3 tpacf_grid;
        dim3 tpacf_block;
        tpacf_block.x = BLOCK_SIZE;
        tpacf_block.y = 1;
        tpacf_grid.x = NUM_SETS * 2 + 1;
        tpacf_grid.y = 1;
        printf("[ORI] Running with tpacf...\n");
        printf("[ORI] tpacf_grid -- %d * %d * %d tpacf_block -- %d * %d * %d \n", 
                tpacf_grid.x, tpacf_grid.y, tpacf_grid.z, tpacf_block.x, tpacf_block.y, tpacf_block.z);
        
        cudaMemcpyToSymbol(dev_binb, binb, (NUM_BINS+1)*sizeof(float));
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_tpacf <<< tpacf_grid, tpacf_block >>> (tpacf_ori_hists, tpacf_ori_x, tpacf_ori_y, tpacf_ori_z, 
                            NUM_SETS, NUM_ELEMENTS, tpacf_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] tpacf took %f ms\n\n", kernel_time);
        serial_time += kernel_time;
    // ---------------------------------------------------------------------------------------


    // PTB
    // ---------------------------------------------------------------------------------
    int tpacf_grid_dim_x = tpacf_grid.x;
    int tpacf_grid_dim_y = tpacf_grid.y;
    int tpacf_block_dim_x = tpacf_block.x;
    int tpacf_block_dim_y = tpacf_block.y;
    tpacf_grid.x = tpacf_blks == 0 ? tpacf_grid_dim_x * tpacf_grid_dim_y : SM_NUM * tpacf_blks;
    tpacf_grid.y = 1;
    tpacf_block.x = tpacf_block_dim_x * tpacf_block_dim_y;
    tpacf_block.y = 1;


	// MIX running 
    // -----------------------------------------------------------------------------------
	if (mixwarp == 1) {
		dim3 mix_grid, mix_block;
        mix_grid.x = (tpacf_grid.x > wmma_grid.x) ? tpacf_grid.x : wmma_grid.x;
        mix_grid.y = 1;
        mix_block.x = tpacf_block.x + wmma_block.x;
        mix_block.y = 1;

        mix_grid.x = SM_NUM;
        mix_block.x = tpacf_block.x + wmma_block.x * 2;

        printf("[PTB] tpacf_grid -- %d * %d * %d tpacf_block -- %d * %d * %d \n", 
            tpacf_grid.x, tpacf_grid.y, tpacf_grid.z, tpacf_block.x, tpacf_block.y, tpacf_block.z);
        printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);
        cudaMemcpyToSymbol(dev_binb, binb, (NUM_BINS+1)*sizeof(float));

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel1 <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c,
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// sgemm parameters
			tpacf_ptb_hists, tpacf_ptb_x, tpacf_ptb_y, tpacf_ptb_z, 
            NUM_SETS, NUM_ELEMENTS, tpacf_grid_dim_x, tpacf_grid_dim_y, tpacf_block_dim_x, tpacf_block_dim_y, 
            tpacf_iter
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
		checkKernelErrors((ptb_tpacf <<< tpacf_grid, tpacf_block, 0, streams[1] >>> (tpacf_ptb_hists, 
                        tpacf_ptb_x, tpacf_ptb_y, tpacf_ptb_z, 
                        NUM_SETS, NUM_ELEMENTS, tpacf_grid_dim_x, tpacf_grid_dim_y, tpacf_block_dim_x, tpacf_block_dim_y, 
                        tpacf_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] tpacf_grid -- %d * %d * %d tpacf_block -- %d * %d * %d \n", 
            tpacf_grid.x, tpacf_grid.y, tpacf_grid.z, tpacf_block.x, tpacf_block.y, tpacf_block.z);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
    	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);

        tpacf_block.x = BLOCK_SIZE;
        tpacf_block.y = 1;
        tpacf_grid.x = NUM_SETS * 2 + 1;
        tpacf_grid.y = 1;

        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_tpacf <<< tpacf_grid, tpacf_block, 0, streams[1] >>> (tpacf_ptb_hists, 
                        tpacf_ptb_x, tpacf_ptb_y, tpacf_ptb_z, 
                        NUM_SETS, NUM_ELEMENTS, 
                        tpacf_iter)));
		
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
    cudaErrCheck(cudaMemcpy(host_tpacf_ori_hists, tpacf_ori_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_tpacf_ptb_hists, tpacf_ptb_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t), cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < NUM_BINS * (NUM_SETS*2+1); i++) {
        unsigned int v1 = host_tpacf_ori_hists[i];
        unsigned int v2 = host_tpacf_ptb_hists[i];
        if (v1 - v2 != 0) {
        errors++;
        if (errors < 5) printf("%u %u\n", v1, v2);
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