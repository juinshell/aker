
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

#include "header/cp_header.h"
#include "kernel/cp_kernel.cu"

#include "header/fft_header.h"
#include "kernel/fft_kernel.cu"

#include "mix_kernel/cp-fft.cu" 

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

    // cp variables
    // ---------------------------------------------------------------------------------------
		int cp_blks = 8;
	    int cp_iter = 1;
        float *atoms = NULL;
		int atomcount = ATOMCOUNT;
		const float gridspacing = 0.1;					// number of atoms to simulate
		dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
		initatoms(&atoms, atomcount, volsize, gridspacing);

		// allocate and initialize the GPU output array
		int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

        float *ori_output;	
		float *ptb_output;
		float *gptb_output;
		cudaErrCheck(cudaMalloc((void**)&ori_output, volmemsz));
		cudaErrCheck(cudaMemset(ori_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
		cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
		cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
		float *host_ori_energy = (float *) malloc(volmemsz);
		float *host_ptb_energy = (float *) malloc(volmemsz);
		float *host_gptb_energy = (float *) malloc(volmemsz);

        dim3 cp_grid, cp_block, ori_cp_grid, ori_cp_block;
        int atomstart = 1;
		int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
		cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
		cp_block.y = BLOCKSIZEY;
		cp_block.z = 1;
		cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
		cp_grid.y = volsize.y / cp_block.y; 
		cp_grid.z = volsize.z / cp_block.z; 
		ori_cp_grid = cp_grid;
		ori_cp_block = cp_block;
		printf("[ORI] Running with cp...\n");
		printf("[ORI] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ori_output, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] cp took %f ms\n\n", kernel_time);

        cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

	// PTB running
    // ---------------------------------------------------------------------------------------
        int solo_ptb_cp_blks = 6;
	    cp_iter = 1;
		int cp_grid_dim_x = cp_grid.x;
		int cp_grid_dim_y = cp_grid.y;
		cp_grid.x = solo_ptb_cp_blks == 0 ? cp_grid_dim_x * cp_grid_dim_y : SM_NUM * solo_ptb_cp_blks;
		cp_grid.y = 1;
		printf("[PTB] Running with cp...\n");
		printf("[PTB] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb2_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ptb_output, 
			cp_grid_dim_x, cp_grid_dim_y, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] cp took %f ms\n\n", kernel_time);

        cudaMemcpy(host_ptb_energy, ptb_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

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


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(768, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_cp_fft_kernel_3_3 <<<mix_kernel_grid, mix_kernel_block>>>(runatoms, 0.1, gptb_output, ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z, 
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cp_grid.x * ori_cp_grid.y * ori_cp_grid.z, fft_gptb_source, 
		ori_fft_grid.x, ori_fft_grid.y, ori_fft_grid.z, ori_fft_block.x, ori_fft_block.y, ori_fft_block.z,
		0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_fft_grid.x * ori_fft_grid.y * ori_fft_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cp_fft 3_3 took %f ms\n\n", kernel_time);
  // ---------------------------------------------------------------------------------------


	// Checking results
    // ---------------------------------------------------------------------------------------
    	cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
	    cudaMemcpy(host_gptb_energy, gptb_output, volmemsz,  cudaMemcpyDeviceToHost);
            
        errors = 0;
        for (int i = 0; i < volsize.x * volsize.y * volsize.z; i++) {
            float v1 = host_ori_energy[i];
            float v2 = host_gptb_energy[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 10) printf("%f %f\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORI VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIG VERSION and GPTB VERSION agree.\n");
        }
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

}
