#include <stdio.h>
#include <float.h>
#include <algorithm>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>


#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
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
#include "file_t/tzgemm_kernel.cu"


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


__global__ void MaxForward(const int nthreads, const float* bottom_data_a,
    const float* bottom_data_b, const int blob_idx, float* top_data,
    float* mask, int iteration) {
	for (int iter_t = 0; iter_t < iteration; iter_t++) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
				i < nthreads; 
				i += blockDim.x * gridDim.x) {
			float maxval = -FLT_MAX;
			int maxidx = -1;
			if (bottom_data_a[i] > bottom_data_b[i]) {
				// only update for very first bottom_data blob (blob_idx == 0)
				if (blob_idx == 0) {
					maxval = bottom_data_a[i];
					top_data[i] = maxval;
					maxidx = blob_idx;
					mask[i] = maxidx;
				}
			} else {
				maxval = bottom_data_b[i];
				top_data[i] = maxval;
				maxidx = blob_idx + 1;
				mask[i] = maxidx;
			}
		}
	}
}


int main(int argc, char* argv[]) {
    int elt_blks = 1;
	int elt_iter = 260;
    if (argc == 3) {
        elt_blks = atoi(argv[1]);
        elt_iter = atoi(argv[2]);
    }

    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));
	cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaErrCheck(cudaStreamCreate(&streams[i]));
    }

	float *bottom0;
	float *bottom1;
	float *top;
	float *max_idx;
	
	int top_n = 16;
	int top_c = 256;
	int top_h = 56;
	int top_w = 56;
	int count = top_n * top_c * top_h * top_w;

	cudaMalloc((void**)&bottom0, count * sizeof(float));
	cudaMalloc((void**)&bottom1, count * sizeof(float));
    cudaMalloc((void**)&top, count * sizeof(float));
    cudaMalloc((void**)&max_idx, count * sizeof(float));

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, bottom0, count));
    curandErrCheck(curandGenerateUniform(gen, bottom1, count));
    curandErrCheck(curandGenerateUniform(gen, top, count));
    curandErrCheck(curandGenerateUniform(gen, max_idx, count));

    dim3 elt_grid;
    dim3 elt_block;

	elt_block.x = 512;
	elt_grid.x = count / 512;
    elt_grid.x = elt_blks == 0 ? count / 512 : 68 * elt_blks;

	printf("[ORI] Running with elt...\n");
    printf("[ORI] elt_grid -- %d * %d elt_block -- %d * %d \n", 
        elt_grid.x, elt_grid.y, elt_block.x, elt_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((MaxForward<<<elt_grid, elt_block>>> (
		count, bottom0, bottom1, 0, top, max_idx, elt_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] elt took %f ms\n\n", kernel_time);

    return 0;
}