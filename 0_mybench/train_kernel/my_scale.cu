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


__global__ void ScaleBiasForward(const int n, const float* in,
    const float* scale, const float* bias,
    const int scale_dim, const int inner_dim, float* out, int iteration) {
    for (int iter_t = 0; iter_t < iteration; iter_t++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                i < n; 
                i += blockDim.x * gridDim.x) {
            const int scale_index = (i / inner_dim) % scale_dim;
            out[i] = in[i] * scale[scale_index] + bias[scale_index];
        }
    }
}


int main(int argc, char* argv[]) {
    int scale_blks = 1;
	int scale_iter = 260;
    if (argc == 3) {
        scale_blks = atoi(argv[1]);
        scale_iter = atoi(argv[2]);
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


    float *bottom;
    float *top;
    float *scale;
    float *bias;

    int top_n = 16;
    int top_c = 64;
    int top_h = 112;
    int top_w = 112;

    int bottom_n = 16;
    int bottom_c = 64;
    int bottom_h = 112;
    int bottom_w = 112;

    int scale_n = 64;
    int bias_n = 64;
    int count = top_n * top_c * top_h * top_w;
    int scale_dim = 64;
    int inner_dim = 12544;

	cudaMalloc((void**)&bottom, bottom_n * bottom_c * bottom_h * bottom_w * sizeof(float));
	cudaMalloc((void**)&top, top_n * top_c * top_h * top_w * sizeof(float));
    cudaMalloc((void**)&scale, scale_n * sizeof(float));
    cudaMalloc((void**)&bias, bias_n * sizeof(float));

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, bottom, bottom_n * bottom_c * bottom_h * bottom_w));
    curandErrCheck(curandGenerateUniform(gen, top, top_n * top_c * top_h * top_w));
    curandErrCheck(curandGenerateUniform(gen, scale, scale_n));
    curandErrCheck(curandGenerateUniform(gen, bias, bias_n));

    dim3 scale_grid;
    dim3 scale_block;

	scale_block.x = 256;
	scale_grid.x = count / 256;
    scale_grid.x = scale_blks == 0 ? count / 256 : 68 * scale_blks;

	printf("[ORI] Running with scale...\n");
    printf("[ORI] scale_grid -- %d * %d scale_block -- %d * %d \n", 
        scale_grid.x, scale_grid.y, scale_block.x, scale_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ScaleBiasForward<<<scale_grid, scale_block>>> (
		count, bottom, scale, bias, scale_dim, inner_dim, top,
        scale_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] scale took %f ms\n\n", kernel_time);

    return 0;
}