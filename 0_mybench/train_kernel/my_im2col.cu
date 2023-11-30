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


// CUDA: check for error after kernel execution and exit loudly if there is one.
// #define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

// // CUDA: use 512 threads per block
// int CAFFE_CUDA_NUM_THREADS = 256;

// // CUDA: number of blocks for threads.
// inline int CAFFE_GET_BLOCKS(int N) {
//   return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
// }


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
	int im_iter = 260;
	int is_caffe = 0;
    if (argc == 3) {
        im_blks = atoi(argv[1]);
        im_iter = atoi(argv[2]);
    }

	float kernel_time;
    float serial_time = 0;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));

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

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, bottom, input_n * input_c * input_h * input_w));
    curandErrCheck(curandGenerateUniform(gen, col_buffer, col_n * col_c * col_h * col_w));

    // im2col_gpu(data_im, input_c, height, width, kernel_h, kernel_w,
    //             pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, data_col);

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
	
	cudaErrCheck(cudaEventRecord(startKERNEL));

	checkKernelErrors((im2col_gpu_kernel<<<im_grid, im_block>>>(
		num_kernels, bottom, input_h, input_w, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
		width_col, col_buffer, im_iter)));

	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] im2col took %f ms\n\n", kernel_time);

    cudaFree(bottom);
    cudaFree(col_buffer);

    return 0;
}