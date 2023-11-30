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


__global__ void im2col_gpu_kernel(int n, float* data_im,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col,
    float* data_col, int iteration) {
	for (int iter_t = 0; iter_t < iteration; iter_t++) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
				i < n; 
				i += blockDim.x * gridDim.x) {
			int h_index = i / width_col;
			int h_col = h_index % height_col;
			int w_col = i % width_col;
			int c_im = h_index / height_col;
			int c_col = c_im * kernel_h * kernel_w;
			int h_offset = h_col * stride_h - pad_h;
			int w_offset = w_col * stride_w - pad_w;
			float* data_col_ptr = data_col;
			data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
			float* data_im_ptr = data_im;
			data_im_ptr += (c_im * height + h_offset) * width + w_offset;
			for (int i = 0; i < kernel_h; ++i) {
				for (int j = 0; j < kernel_w; ++j) {
					int h_im = h_offset + i * dilation_h;
					int w_im = w_offset + j * dilation_w;
					*data_col_ptr =
						(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
						data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
					data_col_ptr += height_col * width_col;
				}
			}
		}
	}
}


int main(int argc, char* argv[]) {
	int im_blks = 1;
	int im_iter = 270;
	int is_caffe = 0;
    if (argc == 3) {
        im_blks = atoi(argv[1]);
        im_iter = atoi(argv[2]);
    }

	// ---------------------------------------------------------------------------------------
		int wmma_blks = 2;
		int wmma_iter = 1;
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
	// ---------------------------------------------------------------------------------------


    // tzgemm variables
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

    int input_n = 16;
	int input_c = 3;
	int input_h = 224;
	int input_w = 224;
	int output_n = 16;
	int output_c = 64;
	int output_h = 112;
	int output_w = 112;
	int col_n = 16;
	int col_c = 147;
	int col_h = 112;
	int col_w = 112;
	float *top;
	float *bottom;
	float *col_buffer;
	cudaMalloc((void**)&bottom, input_n * input_c * input_h * input_w * sizeof(float));
    cudaMalloc((void**)&col_buffer, col_n * col_c * col_h * col_w * sizeof(float));

	int kernel_h = 7;
	int kernel_w = 7;
	int pad_h = 3;
	int pad_w = 3;
	int stride_h = 2;
	int stride_w = 2;
    int dilation_h = 1;
	int dilation_w = 1;

	// curandGenerator_t gen;
    // curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, bottom, input_n * input_c * input_h * input_w));
    curandErrCheck(curandGenerateUniform(gen, col_buffer, col_n * col_c * col_h * col_w));

	int height_col = (input_h + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (input_w + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int num_kernels = input_n * input_c * height_col * width_col;

	dim3 im_grid;
	dim3 im_block;

	im_block.x = 256;
	im_grid.x = int(num_kernels / 256);
	im_grid.x = 68 * 1;

	printf("[ORI] Running with im2col...\n");
    printf("[ORI] im_grid -- %d * %d im_block -- %d * %d \n", 
        im_grid.x, im_grid.y, im_block.x, im_block.y);

	checkKernelErrors((im2col_gpu_kernel<<<im_grid, im_block>>>(
		num_kernels, bottom, input_h, input_w, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
		width_col, col_buffer, im_iter)));
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((im2col_gpu_kernel<<<im_grid, im_block>>>(
		num_kernels, bottom, input_h, input_w, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
		width_col, col_buffer, im_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] im2col took %f ms\n\n", kernel_time);

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

	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, 0, streams[0]>>>(
                            wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));

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
	checkKernelErrors((im2col_gpu_kernel<<<im_grid, im_block, 0, streams[1]>>>(
		num_kernels, bottom, input_h, input_w, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
		width_col, col_buffer, im_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[STREAMP] mix took %f ms\n\n", kernel_time);

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

    cudaFree(bottom);
    cudaFree(col_buffer);

    return 0;
}