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



__global__ void LRNFillScale( int nthreads,  float*  in,
     int num,  int channels,  int height,
     int width,  int size,  float alpha_over_size,
     float k, float*  scale, int iteration) {
    for (int iter_t = 0; iter_t < iteration; iter_t++) {
        CUDA_KERNEL_LOOP(index, nthreads) {
        // find out the local offset
            int w = index % width;
            int h = (index / width) % height;
            int n = index / width / height;
            int offset = (n * channels * height + h) * width + w;
            int step = height * width;
            float*  in_off = in + offset;
            float*  scale_off = scale + offset;
            int head = 0;
            int pre_pad = (size - 1) / 2;
            int post_pad = size - pre_pad - 1;
            float accum_scale = 0;
            // fill the scale at [n, :, h, w]
            // accumulate values
            while (head < post_pad && head < channels) {
                accum_scale += in_off[head * step] * in_off[head * step];
                ++head;
            }
            // both add and subtract
            while (head < channels) {
                accum_scale += in_off[head * step] * in_off[head * step];
                if (head - size >= 0) {
                accum_scale -= in_off[(head - size) * step]
                                * in_off[(head - size) * step];
                }
                scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
                ++head;
            }
            // subtract only
            while (head < channels + post_pad) {
                if (head - size >= 0) {
                accum_scale -= in_off[(head - size) * step]
                                * in_off[(head - size) * step];
                }
                scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
                ++head;
            }
        }
    }
}


__global__ void LRNComputeDiff( int nthreads,
     float*  bottom_data,  float*  top_data,
     float*  scale,  float*  top_diff,
     int num,  int channels,  int height,
     int width,  int size,  float negative_beta,
     float cache_ratio, float*  bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local offset
     int w = index % width;
     int h = (index / width) % height;
     int n = index / width / height;
     int offset = (n * channels * height + h) * width + w;
     int step = height * width;
     float*  bottom_off = bottom_data + offset;
     float*  top_off = top_data + offset;
     float*  scale_off = scale + offset;
     float*  top_diff_off = top_diff + offset;
    float*  bottom_diff_off = bottom_diff + offset;
    int head = 0;
     int pre_pad = size - (size + 1) / 2;
     int post_pad = size - pre_pad - 1;
    float accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step] /
          scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step] *
            top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] =
          top_diff_off[(head - post_pad) * step]
            * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}


int main(int argc, char* argv[]) {
    // 200000
    int lrn_iter = atoi(argv[1]);

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

    float *in;
    float *scale;
    int in_h = 256;
    int in_w = 256;
    int in_c = 4;

    int count = in_h * in_w * in_c;
    int num = 1;
    int size = 1;
    float alpha = 2;
    float k = 5;

    cudaMalloc((void**)&in, count * sizeof(float));
    cudaMalloc((void**)&scale, count * sizeof(float));

    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    // curandErrCheck(curandGenerateUniform(gen, in, count));
    // curandErrCheck(curandGenerateUniform(gen, scale, count));

	#if 1
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

	// curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    // curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
	curandErrCheck(curandGenerateUniform(gen, ori_host_A, MATRIX_M * MATRIX_K));
    curandErrCheck(curandGenerateUniform(gen, ori_host_B, MATRIX_N * MATRIX_K));
	convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_a, ori_host_A, MATRIX_M * MATRIX_K);
    convertFp32ToFp16 <<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_b, ori_host_B, MATRIX_N * MATRIX_K);
	cudaErrCheck(cudaMemset(wmma_ori_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
	cudaErrCheck(cudaMemset(wmma_ptb_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
    // ---------------------------------------------------------------------------------------
	#endif

    dim3 lrn_grid;
    dim3 lrn_block;

    lrn_block.x = 256;
    lrn_grid.x = in_h * in_w * num / 256;

    printf("[ORI] Running with relu...\n");
    printf("[ORI] lrn_grid -- %d * %d lrn_block -- %d * %d \n", 
        lrn_grid.x, lrn_grid.y, lrn_block.x, lrn_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));

    checkKernelErrors((LRNFillScale<<<lrn_grid, lrn_block>>> (
        count, in, num, 
        in_c, in_h, in_w, 
        size, alpha,
        k, scale, lrn_iter
    )));

    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[PTB] relu took %f ms\n\n", kernel_time);

	#if 1
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
    checkKernelErrors((LRNFillScale<<<lrn_grid, lrn_block, 0, streams[1]>>> (
        count, in, num, 
        in_c, in_h, in_w, 
        size, alpha,
        k, scale, lrn_iter
    )));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[STREAMP] mix took %f ms\n\n", kernel_time);

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);
	#endif

    cudaFree(in);
    cudaFree(scale);

    return 0;
}