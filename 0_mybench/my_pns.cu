#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <curand.h>
#include <cublas_v2.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
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


#include "file_t/pns_rand.cu"
#include "file_t/pns_kernel.cu"
#include "header/pns_header.h"


int main(int argc, char** argv) 
{
	int pns_blks = 1;
	int pns_iter = 1;
    if (argc == 3) {
        pns_blks = atoi(argv[1]);
        pns_iter = atoi(argv[2]);
    } 

	// variables
    // ---------------------------------------------------------------------------------------
		float kernel_time;
		cudaEvent_t startKERNEL;
		cudaEvent_t stopKERNEL;
		cudaErrCheck(cudaEventCreate(&startKERNEL));
		cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------


	// pns variables
    // ---------------------------------------------------------------------------------------
		N = 1000;
		s = 200;
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
    // ---------------------------------------------------------------------------------------


	// PTB running
    // ---------------------------------------------------------------------------------------
		int pns_grid_dim_x = pns_grid.x;
		int pns_block_dim_x = pns_block.x;
		pns_grid.x = pns_blks == 0 ? pns_grid_dim_x : 68 * pns_blks;
		pns_block.x = pns_block_dim_x;
		printf("[PTB] Running with pns...\n");
		printf("[PTB] pns_grid -- %d * %d * %d pns_block -- %d * %d * %d\n", 
			pns_grid.x, pns_grid.y, pns_grid.z, pns_block.x, pns_block.y, pns_block.z);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_pns<<< pns_grid, pns_block>>> (
			pns_ptb_places, pns_ptb_vars, pns_ptb_maxs, N, s, 5489, 
			pns_grid_dim_x, pns_block_dim_x, pns_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] pns took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------
	

	cudaErrCheck(cudaMemcpy(host_pns_ori_maxs, pns_ori_maxs, block_num * sizeof(int), cudaMemcpyDeviceToHost));
	cudaErrCheck(cudaMemcpy(host_pns_ori_vars, pns_ori_vars, block_num * sizeof(float), cudaMemcpyDeviceToHost));
	cudaErrCheck(cudaMemcpy(host_pns_ptb_maxs, pns_ptb_maxs, block_num * sizeof(int), cudaMemcpyDeviceToHost));
	cudaErrCheck(cudaMemcpy(host_pns_ptb_vars, pns_ptb_vars, block_num * sizeof(float), cudaMemcpyDeviceToHost));

	// Checking results
    // ---------------------------------------------------------------------------------------
		printf("Checking results...\n");
		int errors = 0;
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
    // ---------------------------------------------------------------------------------------

	// free(h_vars);
	// free(h_maxs);

	// printf("petri N=%d s=%d t=%d\n", N, s, t);
	// printf("mean_vars: %f    var_vars: %f\n", results[0], results[1]);
	// printf("mean_maxs: %f    var_maxs: %f\n", results[2], results[3]);

	return 0;
}
