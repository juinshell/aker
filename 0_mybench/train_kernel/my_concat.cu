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


__global__ void Concat(const int nthreads, const float* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, float* out_data, int iteration) {
    for (int iter_t = 0; iter_t < iteration; iter_t++) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                    i < nthreads; 
                    i += blockDim.x * gridDim.x) {
            const int total_concat_size = concat_size * bottom_concat_axis;
            const int concat_num = i / total_concat_size;
            const int concat_index = i % total_concat_size;
            const int top_index = concat_index +
                (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
            if (forward) {
            out_data[top_index] = in_data[i];
            } else {
            out_data[i] = in_data[top_index];
            }
        }
    }
}


int main(int argc, char* argv[]) {
    int ccat_blks = 1;
	int ccat_iter = 260;
    if (argc == 3) {
        ccat_blks = atoi(argv[1]);
        ccat_iter = atoi(argv[2]);
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

    float *top;
    float *bottom;

    int top_n = 16;
    int top_c = 256;
    int top_h = 28;
    int top_w = 28;

    int bottom_n = 16;
    int bottom_c = 256;
    int bottom_h = 28;
    int bottom_w = 28;

    int top_concat_axis = 256;
    int bottom_concat_axis = 64;
    int concat_input_size_ = 784;
    int num_concats_ = 16;
    int kForward = 1;
    int offset_concat_axis = 0;
    int bottom_concat_size = bottom_concat_axis * concat_input_size_;
    int nthreads = bottom_concat_size * num_concats_;

	cudaMalloc((void**)&top, top_n * top_c * top_h * top_w * sizeof(float));
    cudaMalloc((void**)&bottom, bottom_n * bottom_c * bottom_h * bottom_w * sizeof(float));

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, top, top_n * top_c * top_h * top_w));
    curandErrCheck(curandGenerateUniform(gen, bottom, bottom_n * bottom_c * bottom_h * bottom_w));

    dim3 ccat_grid;
    dim3 ccat_block;

	ccat_block.x = 256;
	ccat_grid.x = nthreads / 256;
    ccat_grid.x = ccat_blks == 0 ? nthreads / 256 : 68 * ccat_blks;

	printf("[ORI] Running with ccat...\n");
    printf("[ORI] ccat_grid -- %d * %d ccat_block -- %d * %d \n", 
        ccat_grid.x, ccat_grid.y, ccat_block.x, ccat_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((Concat<<<ccat_grid, ccat_block>>> (
        nthreads, bottom, kForward, num_concats_, concat_input_size_,
        top_concat_axis, bottom_concat_axis, offset_concat_axis, top,
		ccat_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] ccat took %f ms\n\n", kernel_time);

    return 0;
}