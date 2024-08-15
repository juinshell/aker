#include <stdio.h>
// #include <limits>
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


__global__ void MaxPoolForward(const int nthreads,
    const float* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data, float* mask, float* top_mask, int iteration) {
	for (int iter_t = 0; iter_t < iteration; iter_t++) {
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
						i < nthreads; 
						i += blockDim.x * gridDim.x) {
			const int pw = i % pooled_width;
			const int ph = (i / pooled_width) % pooled_height;
			const int c = (i / pooled_width / pooled_height) % channels;
			const int n = i / pooled_width / pooled_height / channels;
			int hstart = ph * stride_h - pad_h;
			int wstart = pw * stride_w - pad_w;
			const int hend = min(hstart + kernel_h, height);
			const int wend = min(wstart + kernel_w, width);
			hstart = max(hstart, 0);
			wstart = max(wstart, 0);
			float maxval = -FLT_MAX;
			int maxidx = -1;
			const float* const bottom_slice =
				bottom_data + (n * channels + c) * height * width;
			for (int h = hstart; h < hend; ++h) {
				for (int w = wstart; w < wend; ++w) {
					if (bottom_slice[h * width + w] > maxval) {
						maxidx = h * width + w;
						maxval = bottom_slice[maxidx];
					}
				}
			}
			top_data[i] = maxval;
			if (mask) {
				mask[i] = maxidx;
			} else {
				top_mask[i] = maxidx;
			}
		}
	}
}


__global__ void MaxPoolBackward(const int nthreads, const float* const top_diff,
    const int* const mask, const float* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, float* const bottom_diff) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    // find out the local i
    // find out the local offset
    const int w = i % width;
    const int h = (i / width) % height;
    const int c = (i / width / height) % channels;
    const int n = i / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    float gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const float* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const float* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[i] = gradient;
  }
}


// MaxPoolForward<float><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//     count, bottom_data, bottom[0]->num(), channels_,
//     height_, width_, pooled_height_, pooled_width_, kernel_h_,
//     kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
//     mask, top_mask);


int main(int argc, char* argv[]) {
    int pool_blks = 1;
	int pool_iter = 260;
    if (argc == 3) {
        pool_blks = atoi(argv[1]);
        pool_iter = atoi(argv[2]);
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
	float *max_idx;
	float *top_mask = NULL;

	int bottom_n = 16;
	int bottom_c = 64;
	int bottom_h = 112;
	int bottom_w = 112;
	int top_n = 16;
	int top_c = 64;
	int top_h = 56;
	int top_w = 56;

	int kernel_h = 3;
	int kernel_w = 3;
	int stride_h = 2;
	int stride_w = 2;
	int pad_h = 0;
	int pad_w = 0;
	int count = top_n * top_c * top_h * top_w;

	cudaMalloc((void**)&bottom, bottom_n * bottom_c * bottom_h * bottom_w * sizeof(float));
	cudaMalloc((void**)&top, top_n * top_c * top_h * top_w * sizeof(float));
	cudaMalloc((void**)&max_idx, top_n * top_c * top_h * top_w * sizeof(float));

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, bottom, bottom_n * bottom_c * bottom_h * bottom_w));
    curandErrCheck(curandGenerateUniform(gen, top, top_n * top_c * top_h * top_w));
    curandErrCheck(curandGenerateUniform(gen, max_idx, top_n * top_c * top_h * top_w));

    dim3 pool_grid;
    dim3 pool_block;

	pool_block.x = 256;
	pool_grid.x = count / 256;
    pool_grid.x = pool_blks == 0 ? count / 256 : 68 * pool_blks;

	printf("[ORI] Running with pool...\n");
    printf("[ORI] pool_grid -- %d * %d pool_block -- %d * %d \n", 
        pool_grid.x, pool_grid.y, pool_block.x, pool_block.y);
	
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((MaxPoolForward<<<pool_grid, pool_block>>> (
		count, bottom,
		bottom_n, bottom_c, bottom_h, bottom_w,
		top_h, top_w,
		kernel_h, kernel_w,
		stride_h, stride_w,
		pad_h, pad_w,
		top, max_idx, top_mask, pool_iter
	)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] pool took %f ms\n\n", kernel_time);

    return 0;
}