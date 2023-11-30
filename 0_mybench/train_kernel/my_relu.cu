#include <stdio.h>
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


__global__ void ReLUForward(int n, float* in, float* out,
    float negative_slope, int iteration) {
	for (int iter_t = 0; iter_t < iteration; iter_t++) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
				i < n; 
				i += blockDim.x * gridDim.x) {
			out[i] = in[i] > 0 ? in[i] : in[i] * negative_slope;
		}
	}
}


__global__ void ReLUBackward(int n, float* in_diff,
    float* in_data, float* out_diff, float negative_slope) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                    i < n; 
                    i += blockDim.x * gridDim.x) {
        out_diff[i] = in_diff[i] * ((in_data[i] > 0)
            + (in_data[i] <= 0) * negative_slope);
    }
}


int main(int argc, char* argv[]) {
    int relu_blks = 1;
	int relu_iter = 260;
    if (argc == 3) {
        relu_blks = atoi(argv[1]);
        relu_iter = atoi(argv[2]);
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


    float *in;
    float *out;
    int in_h = 1024 * 4;
    int in_w = 1024 * 4;
    int in_c = 4;
    int count = in_h * in_w * in_c;

	cudaMalloc((void**)&in, count * sizeof(float));
    cudaMalloc((void**)&out, count * sizeof(float));

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, in, count));
    curandErrCheck(curandGenerateUniform(gen, out, count));

    dim3 relu_grid;
    dim3 relu_block;

	relu_block.x = 256;
	relu_grid.x = count / 256;
    relu_grid.x = relu_blks == 0 ? count / 512 : 68 * relu_blks;

	printf("[ORI] Running with relu...\n");
    printf("[ORI] relu_grid -- %d * %d relu_block -- %d * %d \n", 
        relu_grid.x, relu_grid.y, relu_block.x, relu_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ReLUForward<<<relu_grid, relu_block>>> (
		count, in, out,
		-1, relu_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] relu took %f ms\n\n", kernel_time);

    cudaFree(in);
    cudaFree(out);

    return 0;
}