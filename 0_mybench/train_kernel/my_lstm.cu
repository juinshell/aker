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
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)



__global__ void LSTMUnitForward(int nthreads, int dim,
    float* C_prev, float* X, float* cont,
    float* C, float* H, int iteration) {
    
    for (int iter_t = 0; iter_t < iteration; iter_t++) {
        CUDA_KERNEL_LOOP(index, nthreads) {
            int n = index / dim;
            int d = index % dim;
            float* X_offset = X + 4 * dim * n;
            float i = X_offset[d];
            float f = X_offset[1 * dim + d];
            float o = X_offset[2 * dim + d];
            float g = X_offset[3 * dim + d];
            float c_prev = C_prev[index];
            float c = cont[n] * f * c_prev + i * g;
            C[index] = c;
            float tanh_c = tanh(c);
            H[index] = o * tanh_c;
        }
    }
}


__global__ void LSTMUnitBackward(int nthreads, int dim,
    float* C_prev, float* X, float* C, float* H,
    float* cont, float* C_diff, float* H_diff,
    float* C_prev_diff, float* X_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / dim;
    int d = index % dim;
    float* X_offset = X + 4 * dim * n;
    float i = X_offset[d];
    float f = X_offset[1 * dim + d];
    float o = X_offset[2 * dim + d];
    float g = X_offset[3 * dim + d];
    float c_prev = C_prev[index];
    float c = C[index];
    float tanh_c = tanh(c);
    float* c_prev_diff = C_prev_diff + index;
    float* X_diff_offset = X_diff + 4 * dim * n;
    float* i_diff = X_diff_offset + d;
    float* f_diff = X_diff_offset + 1 * dim + d;
    float* o_diff = X_diff_offset + 2 * dim + d;
    float* g_diff = X_diff_offset + 3 * dim + d;
    float c_term_diff =
        C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
    float cont_n = cont[n];
    *c_prev_diff = cont_n * c_term_diff * f;
    *i_diff = c_term_diff * g;
    *f_diff = cont_n * c_term_diff * c_prev;
    *o_diff = H_diff[index] * tanh_c;
    *g_diff = c_term_diff * i;
  }
}


int main(int argc, char* argv[]) {
    // 300000
    int lstm_iter = atoi(argv[1]);

	// ---------------------------------------------------------------------------------------

    int wmma_blks = 2;
    int wmma_iter = 1800;
    int M_INPUT = 16 * 8 * 32;
	int N_INPUT = 16 * 8 * 24;
	int K_INPUT = 16 * 8 * 6;

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

    
    float *C_prev;
    float *X;
    float *cont;
    float *C;
    float *H;

    int in_h = 256;
    int in_w = 256;
    int in_c = 4;

    int count = in_h * in_w * in_c;

    cudaMalloc((void**)&C_prev, count * sizeof(float));
    cudaMalloc((void**)&C, count * sizeof(float));
    cudaMalloc((void**)&H, count * sizeof(float));
    cudaMalloc((void**)&X, count * sizeof(float));
    cudaMalloc((void**)&cont, count * sizeof(float));

	// curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    // curandErrCheck(curandGenerateUniform(gen, C_prev, count));
    // curandErrCheck(curandGenerateUniform(gen, C, count));
    // curandErrCheck(curandGenerateUniform(gen, H, count));
    // curandErrCheck(curandGenerateUniform(gen, X, count));
    // curandErrCheck(curandGenerateUniform(gen, cont, count));


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


    dim3 lstm_grid;
    dim3 lstm_block;

    lstm_block.x = 256;
    lstm_grid.x = count / 256;

    printf("[ORI] Running with relu...\n");
    printf("[ORI] lstm_grid -- %d * %d lstm_block -- %d * %d \n", 
        lstm_grid.x, lstm_grid.y, lstm_block.x, lstm_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((LSTMUnitForward<<<lstm_grid, lstm_block>>> (
        count, 1,
        C_prev, X, cont,
        C, H, lstm_iter
    )));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[PTB] lstm took %f ms\n\n", kernel_time);

    serial_time += kernel_time;

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

    // int SHMEM_SZ = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
	// cudaErrCheck(cudaFuncSetAttribute(
	// 	ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	// SHMEM_SZ = 0;

    printf("[PTB] Running with tzgemm...\n");
    printf("[PTB] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(
                            wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[PTB] tzgemm took %f ms\n", kernel_time);
    serial_time += kernel_time;

    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(
        wmma_ori_a, wmma_ori_b, wmma_ori_c, 
        MATRIX_M, MATRIX_N, MATRIX_K,
        wmma_grid_dim_x, wmma_block_dim_x, wmma_iter
    )));
    checkKernelErrors((LSTMUnitForward<<<lstm_grid, lstm_block, 0, streams[1]>>> (
        count, 1,
        C_prev, X, cont,
        C, H, lstm_iter
    )));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[STREAMP] mix took %f ms\n\n", kernel_time);

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

    return 0;
}