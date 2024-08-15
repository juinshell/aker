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


__global__ void mul_kernel(const int n, const float* a, const float* b, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                    i < n; 
                    i += blockDim.x * gridDim.x) {
        y[i] = a[i] * b[i];
    }
}


__global__ void saxpy(int n, float a, float* x, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                    i < n; 
                    i += blockDim.x * gridDim.x) {
        y[i] = a*x[i] + y[i];
    }
}


__global__ void add_scalar_kernel(const int n, const float alpha, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                        i < n; 
                        i += blockDim.x * gridDim.x) {
        y[i] += alpha;
    }
}


__global__ void sqrt_kernel(const int n, const float* a, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                            i < n; 
                            i += blockDim.x * gridDim.x) {
        y[i] = sqrt(a[i]);
    }
}


__global__ void div_kernel(const int n, const float* a,
    const float* b, float* y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                                i < n; 
                                i += blockDim.x * gridDim.x) {
        y[i] = a[i] / b[i];
    }
}


__global__ void sgemv_kernel(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y, 
                            int iteration){
    for (int iter_t = 0; iter_t < iteration; iter_t++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
                                i < m; 
                                i += blockDim.x * gridDim.x) {
            float resY = 0;
            for (int j = 0; j < n; j++) {
                resY += A[i + j * lda] * x[j];
            }
            y[i] = alpha * resY + beta * y[i];
        }
    }
}


// void mysgemv_gpu_v1(int m, int n, float alpha, float *A, int lda, float *x, float beta, float *y){
//     dim3 grid( CEIL_DIV(m, DIM_M), 1 );
//     dim3 threads( DIM_M, 1 );
//     sgemv_kernel_v1<<<grid, threads>>>(m, n, alpha, A, lda, x, beta, y);
// }



int main(int argc, char* argv[]) {
    int bn_blks = 1;
	int bn_iter = 260;
    if (argc == 3) {
        bn_blks = atoi(argv[1]);
        bn_iter = atoi(argv[2]);
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

    float *matrix_a;
    float *vector_x;
    float *vector_y;

    int matrix_h = 128*68*2*8;
    int matrix_w = 32;
    cudaMalloc((void**)&matrix_a, matrix_h * matrix_w * sizeof(float));
    cudaMalloc((void**)&vector_x, matrix_h * sizeof(float));
    cudaMalloc((void**)&vector_y, matrix_h * sizeof(float));

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, matrix_a, matrix_h * matrix_w));
    curandErrCheck(curandGenerateUniform(gen, vector_x, matrix_h));
    curandErrCheck(curandGenerateUniform(gen, vector_y, matrix_h));

    dim3 bn_grid;
    dim3 bn_block;

	bn_block.x = 128;
	bn_grid.x = matrix_h / 128;
    bn_grid.x = bn_blks == 0 ? matrix_h / 128 : 68 * bn_blks;

	printf("[ORI] Running with bn...\n");
    printf("[ORI] bn_grid -- %d * %d bn_block -- %d * %d \n", 
        bn_grid.x, bn_grid.y, bn_block.x, bn_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((sgemv_kernel<<<bn_grid, bn_block>>> (
		matrix_h, matrix_w, 0.1, matrix_a,
        matrix_h, vector_x, 0.2, vector_y, bn_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] bn took %f ms\n\n", kernel_time);

    return 0;
}