
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <malloc.h>
#include <sys/time.h>
using namespace nvcuda; 

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

#include "header/fft_header.h"
#include "kernel/fft_kernel.cu"

#include "header/sgemm_header.h"
#include "kernel/sgemm_kernel.cu"

#include "mix_kernel/fft-sgemm.cu" 

int main(int argc, char* argv[]) {
    int errors = 0;

	  // variables
    // ---------------------------------------------------------------------------------------
		float kernel_time;
		cudaEvent_t startKERNEL;
		cudaEvent_t stopKERNEL;
		cudaErrCheck(cudaEventCreate(&startKERNEL));
		cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    // fft variables
    // ---------------------------------------------------------------------------------------
		//8*1024*1024;
        int fft_blks = 3;
	    int fft_iter = 1;
		int n_bytes = FFT_N * FFT_B * sizeof(float2);
		int nthreads = FFT_T;
		srand(54321);

		float *host_shared_source =(float *)malloc(n_bytes);  
		float2 *source    = (float2 *)malloc( n_bytes );
		float2 *host_fft_ori_result    = (float2 *)malloc( n_bytes );
		float2 *host_fft_ptb_result    = (float2 *)malloc( n_bytes );
		float2 *host_fft_gptb_result	= (float2 *)malloc( n_bytes );

		for(int b=0; b<FFT_B;b++) {	
			for( int i = 0; i < FFT_N; i++ ) {
				source[b*FFT_N+i].x = (rand()/(float)RAND_MAX)*2-1;
				source[b*FFT_N+i].y = (rand()/(float)RAND_MAX)*2-1;
			}
		}

		// allocate device memory
		float2 *fft_ori_source;
		float *fft_ori_shared_source;
		cudaMalloc((void**) &fft_ori_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ori_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_ori_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ori_source, source, n_bytes, cudaMemcpyHostToDevice);

		float2 *fft_ptb_source;
		float *fft_ptb_shared_source;
		cudaMalloc((void**) &fft_ptb_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ptb_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_ptb_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_ptb_source, source, n_bytes, cudaMemcpyHostToDevice);

		// gptb
		float2 *fft_gptb_source;
		float *fft_gptb_shared_source;
		cudaMalloc((void**) &fft_gptb_shared_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_gptb_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
		cudaMalloc((void**) &fft_gptb_source, n_bytes);
		// copy host memory to device
		cudaMemcpy(fft_gptb_source, source, n_bytes, cudaMemcpyHostToDevice);
    // ---------------------------------------------------------------------------------------

	// SOLO running
    // ---------------------------------------------------------------------------------------
		dim3 fft_grid, ori_fft_grid;
		dim3 fft_block, ori_fft_block;
		fft_grid.x = FFT_B;
		fft_block.x = nthreads;
		ori_fft_grid = fft_grid;
		ori_fft_block = fft_block;

		printf("[ORI] Running with fft...\n");
		printf("[ORI] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\n", 
			fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_fft<<<fft_grid, fft_block>>>(fft_ori_source, fft_iter))); 	
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] fft took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------
 
	// PTB running
    // ---------------------------------------------------------------------------------------
		int fft_grid_dim_x = fft_grid.x;
		int fft_block_dim_x = fft_block.x;
		fft_grid.x = fft_blks == 0 ? fft_grid_dim_x : SM_NUM * fft_blks;
		fft_block.x = fft_block_dim_x;
		printf("[PTB] Running with fft...\n");
		printf("[PTB] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\n", 
			fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_fft<<<fft_grid, fft_block>>>(fft_ptb_source, fft_grid_dim_x, fft_block_dim_x, fft_iter))); 	
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] fft took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // sgemm variables
    // ---------------------------------------------------------------------------------------
        int sgemm_blks = 4;
        int sgemm_iter = 1;
        float *sgemm_ori_a;
        float *sgemm_ori_b;
        float *sgemm_ori_c;
        float *sgemm_ptb_a;
        float *sgemm_ptb_b;
        float *sgemm_ptb_c;
        float *sgemm_gptb_a;
        float *sgemm_gptb_b;
        float *sgemm_gptb_c;
        float *host_sgemm_ori_c;
        float *host_sgemm_ptb_c;
        float *host_sgemm_gptb_c;
        

        // parallel experiment
        int NORMAL_M = 4096;
        int NORMAL_N = 4128;
        int NORMAL_K = 4064;

        NORMAL_M = (NORMAL_M / 10) * sgemm_iter;

        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_c, NORMAL_M * NORMAL_N * sizeof(float)));

        host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        host_sgemm_ptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        host_sgemm_gptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));

        curandGenerator_t sgemm_gen;
        curandErrCheck(curandCreateGenerator(&sgemm_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(sgemm_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
        cudaErrCheck(cudaMemcpy(sgemm_ptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_ptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_gptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_gptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        curandErrCheck(curandDestroyGenerator(sgemm_gen));
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 sgemm_grid, ori_sgemm_grid;
        dim3 sgemm_block, ori_sgemm_block;
        sgemm_block.x = TILE_N;
        sgemm_block.y = TILE_TB_HEIGHT;
        sgemm_grid.x = NORMAL_M/TILE_M;
        sgemm_grid.y = NORMAL_N/TILE_N;
        ori_sgemm_grid = sgemm_grid;
        ori_sgemm_block = sgemm_block;
        printf("[ORI] Running with sgemm...\n");
        printf("[ORI] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block >>> (
                    sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
                    NORMAL_M, NORMAL_N, NORMAL_K, 1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] sgemm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        int sgemm_grid_dim_x = sgemm_grid.x;
        int sgemm_grid_dim_y = sgemm_grid.y;
        // int sgemm_block_dim_x = sgemm_block.x;
        // int sgemm_block_dim_y = sgemm_block.y;
        sgemm_grid.x = sgemm_grid_dim_x * sgemm_grid_dim_y;
        sgemm_grid.x = sgemm_blks == 0 ? sgemm_grid_dim_x * sgemm_grid_dim_y : 68 * sgemm_blks;
        sgemm_grid.y = 1;
        // sgemm_block.x = sgemm_block_dim_x * sgemm_block_dim_y;
        // sgemm_block.y = 1;
        printf("[PTB] Running with sgemm...\n");
        printf("[PTB] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb2_sgemm <<< sgemm_grid, sgemm_block >>> (
                    sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
                    NORMAL_M, NORMAL_N, NORMAL_K,
                    sgemm_grid_dim_x, sgemm_grid_dim_y, 
                    1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] sgemm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(1024, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_fft_sgemm_kernel_4_4 <<<mix_kernel_grid, mix_kernel_block>>>(fft_gptb_source, 
		ori_fft_grid.x, ori_fft_grid.y, ori_fft_grid.z, ori_fft_block.x, ori_fft_block.y, ori_fft_block.z,
		0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_fft_grid.x * ori_fft_grid.y * ori_fft_grid.z, sgemm_gptb_a, sgemm_gptb_b, sgemm_gptb_c, NORMAL_M, NORMAL_N, NORMAL_K,
            ori_sgemm_grid.x, ori_sgemm_grid.y, ori_sgemm_grid.z, ori_sgemm_block.x, ori_sgemm_block.y, ori_sgemm_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] fft_sgemm 4_4 took %f ms\n\n", kernel_time);
  // ---------------------------------------------------------------------------------------


    // ---------------------------------------------------------------------------------------
	cudaMemcpy(host_fft_ori_result, fft_ori_source, n_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_fft_gptb_result, fft_gptb_source, n_bytes, cudaMemcpyDeviceToHost);
    errors = 0;
	for (int i = 0; i < FFT_N * FFT_B; i++) {
		float v1 = host_fft_ori_result[i].x;
		float v2 = host_fft_gptb_result[i].x;
		if (fabs(v1 - v2) > 0.001f) {
			errors++;
			if (errors < 10) printf("%f %f\n", v1, v2);
		}
		if (i < 3) printf("%d %f %f\n", i, v1, v2);

		v1 = host_fft_ori_result[i].y;
		v2 = host_fft_gptb_result[i].y;
		if (fabs(v1 - v2) > 0.001f) {
			errors++;
			if (errors < 10) printf("%f %f\n", v1, v2);
		}
	}
	if (errors > 0) {
		printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
	}
	else {
		printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
	}
    // ---------------------------------------------------------------------------------------

    // Checking results
    // ---------------------------------------------------------------------------------------
        cudaErrCheck(cudaMemcpy(host_sgemm_ori_c, sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_sgemm_gptb_c, sgemm_gptb_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));

        errors = 0;
        for (int i = 0; i < NORMAL_M * NORMAL_N; i++) {
            float v1 = host_sgemm_ori_c[i];
            float v2 = host_sgemm_gptb_c[i];
            if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
        }
    // ---------------------------------------------------------------------------------------


}
