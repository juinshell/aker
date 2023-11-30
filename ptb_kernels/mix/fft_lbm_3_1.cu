
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

#include "header/lbm_header.h"
#include "kernel/lbm_kernel.cu"

#include "mix_kernel/fft-lbm.cu" 

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


    // lbm variables
    // ---------------------------------------------------------------------------------------
        int lbm_blks = 1;
        int lbm_iter = 1;
        float *lbm_ori_src;
        float *lbm_ori_dst;
        float *lbm_ptb_src;
        float *lbm_ptb_dst;
        float *lbm_gptb_src;
        float *lbm_gptb_dst;
        float *host_lbm_ori_dst;
        float *host_lbm_ptb_dst;
        float *host_lbm_gptb_dst;

        const size_t size = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float) + 2 * TOTAL_MARGIN * sizeof(float);

        host_lbm_ori_dst = (float *)malloc(size);
        host_lbm_ptb_dst = (float *)malloc(size);
        host_lbm_gptb_dst = (float *)malloc(size);
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_gptb_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_gptb_dst, size));

        curandGenerator_t lbm_gen;
        curandErrCheck(curandCreateGenerator(&lbm_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(lbm_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_src, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_dst, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        cudaErrCheck(cudaMemcpy(lbm_ptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_ptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_gptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_gptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
        lbm_ori_src += REAL_MARGIN;
        lbm_ori_dst += REAL_MARGIN;
        lbm_ptb_src += REAL_MARGIN;
        lbm_ptb_dst += REAL_MARGIN;
        lbm_gptb_src += REAL_MARGIN;
        lbm_gptb_dst += REAL_MARGIN;

    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 lbm_block, lbm_grid, ori_lbm_block, ori_lbm_grid;
        lbm_block.x = SIZE_X;
        lbm_grid.x = SIZE_Y;
        lbm_grid.y = SIZE_Z;
        lbm_block.y = lbm_block.z = lbm_grid.z = 1;
        ori_lbm_block = lbm_block;
        ori_lbm_grid = lbm_grid;
        printf("[ORI] Running with lbm...\n");
        printf("[ORI] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_lbm<<<lbm_grid, lbm_block>>>(lbm_ori_src, lbm_ori_dst, lbm_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] lbm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    // PTB running
    // ---------------------------------------------------------------------------------------
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
        printf("[PTB] Running with lbm...\n");
        printf("[PTB] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_lbm<<<lbm_grid, lbm_block>>>(lbm_ptb_src, lbm_ptb_dst,
            lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
            lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z, lbm_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] lbm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(512, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_fft_lbm_kernel_3_1 <<<mix_kernel_grid, mix_kernel_block>>>(fft_gptb_source, 
		ori_fft_grid.x, ori_fft_grid.y, ori_fft_grid.z, ori_fft_block.x, ori_fft_block.y, ori_fft_block.z,
		0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_fft_grid.x * ori_fft_grid.y * ori_fft_grid.z, lbm_gptb_src, lbm_gptb_dst,
            ori_lbm_grid.x, ori_lbm_grid.y, ori_lbm_grid.z, ori_lbm_block.x, ori_lbm_block.y, ori_lbm_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_lbm_grid.x * ori_lbm_grid.y * ori_lbm_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] fft_lbm 3_1 took %f ms\n\n", kernel_time);
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

    // ---------------------------------------------------------------------------------------
        lbm_ori_src -= REAL_MARGIN;
        lbm_ori_dst -= REAL_MARGIN;
        lbm_ptb_src -= REAL_MARGIN;
        lbm_ptb_dst -= REAL_MARGIN;
        lbm_gptb_src -= REAL_MARGIN;
        lbm_gptb_dst -= REAL_MARGIN;
        cudaErrCheck(cudaMemcpy(host_lbm_ori_dst, lbm_ori_dst, size, cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_lbm_gptb_dst, lbm_gptb_dst, size, cudaMemcpyDeviceToHost));
        errors = 0;
        for (int i = 0; i < TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN; i++) {
            float v1 = host_lbm_ori_dst[i];
            float v2 = host_lbm_gptb_dst[i];
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
