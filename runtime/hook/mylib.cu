#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cudnn.h>
#include <sstream>
#include <mma.h>
#include <iostream>
#include <curand.h>
using namespace nvcuda;

// cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
// {
// cudaError_t (*lcudaMemcpy) ( void*, const void*, size_t, cudaMemcpyKind) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind  ))dlsym(RTLD_NEXT, "cudaMemcpy");
//     printf("cudaMemcpy hooked\n");
//     return lcudaMemcpy( dst, src, count, kind );
// }

// cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t str )
// {
// cudaError_t (*lcudaMemcpyAsync) ( void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = (cudaError_t (*) ( void* , const void* , size_t , cudaMemcpyKind, cudaStream_t   ))dlsym(RTLD_NEXT, "cudaMemcpyAsync");
//     printf("cudaMemcpyAsync hooked\n");
//     return lcudaMemcpyAsync( dst, src, count, kind, str );
// }

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#define CUDNN_SAFE_CALL(func)                                                                      \
    do                                                                                             \
    {                                                                                              \
        cudnnStatus_t e = (func);                                                                  \
        if (e != CUDNN_STATUS_SUCCESS)                                                             \
        {                                                                                          \
            const char* msg = cudnnGetErrorString(e);                                              \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
#define CUBLAS_SAFE_CALL(func)                                                                     \
    do                                                                                             \
    {                                                                                              \
        cublasStatus_t e = (func);                                                                 \
        if (e != CUBLAS_STATUS_SUCCESS)                                                            \
        {                                                                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #func " failed with error"                                 \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << e;    \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)
   #define CUDA_SAFE_CALL(x)                                                                          \
    do                                                                                             \
    {                                                                                              \
        cudaError_t result = (x);                                                                  \
        if (result != cudaSuccess)                                                                 \
        {                                                                                          \
            const char* msg = cudaGetErrorString(result);                                          \
            std::stringstream safe_call_ss;                                                        \
            safe_call_ss << "\nerror: " #x " failed with error"                                    \
                         << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg;  \
            throw std::runtime_error(safe_call_ss.str());                                          \
        }                                                                                          \
    } while (0)

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

// gemm
#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// #define M_TILES (8 * 32)
// #define N_TILES (8 * 24)
// #define K_TILES (8 * 6)
// #define M_GLOBAL (WMMA_M * M_TILES)
// #define N_GLOBAL (WMMA_N * N_TILES)
// #define K_GLOBAL (WMMA_K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// With only 64 Kb shared memory available, we can fit two 8-tile chunks of
// the A and B matrix data, that are 16 * 16 * 8 * 8 * 2 = 32 Kb each
// (i.e. two 8x8 arrays of tiles of 16x16 half-typed elements per CTA).
// But we cannot account the 8 Kb total skew overhead, without which the
// performance would be severely impacted. So we choose to reduce the chunk size
// in half, i.e. the amount of A and B matrix data we cache in shared memory.
// Accordingly, this doubles the number of outer iterations across the global WMMA_K
// dimension, which only slightly impacts the performance.
#define CHUNK_K 4

#define CHUNK_LINE_BYTES (CHUNK_K * WMMA_K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 2
#define WARP_ROW_TILES 2
#define WARP_COL_TILES 2

// Implementation constants.
#define WARPS_PER_BLOCK (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL
#define SHMEM_STRIDE (WMMA_N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (WMMA_N * WARP_ROW_TILES)

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.
#define SKEW_HALF 8

const float alpha_g = 1.1f;
const float beta_g = 0;

#ifndef SM_NUM
#define SM_NUM 68
#endif
#include <cassert>
#define WMMA_GRID_DIM (SM_NUM * 2)
#define WMMA_GRID_DIM2 (SM_NUM * 2)

__global__ void im2col_gpu_kernel(int n, float* data_im,
    int height, int width, int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col,
    float* data_col, int data_im_size, int data_col_size) {
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
                    if (h_col >= height_col || w_col >= width_col || h_col < 0 || w_col < 0 || (data_col_ptr - data_col) >= data_col_size) {
                        // printf("h_col: %d, w_col: %d, height_col: %d, width_col: %d, data_col_ptr - data_col: %d\n", h_col, w_col, height_col, width_col, data_col_ptr - data_col);
                        continue;
                    }
					*data_col_ptr =
						(h_im >= 0 && w_im >= 0 && h_im < height && w_im < width && (i * dilation_h * width + j * dilation_w < data_im_size)) ?
						data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
					data_col_ptr += height_col * width_col;
				}
			}
		}
}

__global__ void ptb_tzgemm(half *A, half *B, float *C, 
		// float alpha, float beta,
		int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
		int grid_dimension_x, int block_dimension_x) {

	__shared__ half shmem[BLOCK_COL_TILES * WMMA_M * 2][CHUNK_K * WMMA_K + SKEW_HALF];
	// extern __shared__ half shmem[][CHUNK_K * WMMA_K + SKEW_HALF];

	const unsigned int N_TILES = N_GLOBAL / WMMA_N;
	const unsigned int K_TILES = K_GLOBAL / WMMA_K;
	// const unsigned int M_TILES = M_GLOBAL / WMMA_M;

	float alpha = alpha_g;
	float beta = beta_g;

	unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

	// Warp and lane identification.
	const unsigned int warpId = thread_id_x / WARP_SIZE;
	const unsigned int laneId = thread_id_x % WARP_SIZE;

	// Offset in shared memory from which the B matrix is stored.
	const size_t shmem_idx_b_off = BLOCK_COL_TILES * WMMA_M;
	// This pointer is used to access the C and D matrix tiles this warp computes.
	float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
								(warpId / 2) * SHMEM_STRIDE * WMMA_M * 2 +
								(warpId % 2) * SHMEM_OFFSET;

	// This pointer is used to stream the C and D matrices block-wide tile to and
	// from shared memory.
	float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * WMMA_M;

	// Adjust the beta scaler, as it'll be multiplied by alpha at the end of
	// each tile computation. Technically this is not generally correct (may
	// result in a loss of precision). Zero still needs to be specially handled
	// though.
	beta /= alpha;

	// Each CTA slides along the 128 x 128 tiles from the top left corner of the
	// matrix to the right and down, and selects the next tile to compute. Once
	// there's no such tile, all warps in this CTA exit.
	for (;; block_pos += gridDim.x) {
		if (block_pos >= grid_dimension_x) {
            return;
        }

		const unsigned int block_tile_i =
			((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
		const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;
		// This warp's pointer to the C matrix data to copy memory from to shared
		// memory.
		const size_t gmem_idx =
			(block_tile_i + warpId) * WMMA_M * GLOBAL_MEM_STRIDE + block_tile_j * WMMA_N;


        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c[WARP_COL_TILES][WARP_ROW_TILES];
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                wmma::fill_fragment(c[i][j], 0.0f);
            }
        }

        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const half *warp_ptr = 
            warpId < (WARPS_PER_BLOCK / 2) 
                ? (&A[block_tile_i * WMMA_M * K_GLOBAL] + WMMA_M * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                : (&B[block_tile_j * WMMA_N * K_GLOBAL] + WMMA_N * K_GLOBAL * (warpId % (WARPS_PER_BLOCK / 2)) * 2);

        // Go through the global K dimension by a fixed step at a time.
        #pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, 
            // the rest copy the B matrix.
            size_t shmem_idx =
                warpId < (WARPS_PER_BLOCK / 2)
                    ? (WMMA_M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                    : (WMMA_N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * WMMA_K + (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) 
                + (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++) {
                // Copy 16 bytes at once in each lane.
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) =
                    *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr =
                    (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }

            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
            #pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b[WARP_ROW_TILES];

                #pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++) {
                    size_t shmem_idx_a = (warpId / 2) * WMMA_M * 2 + (i * WMMA_M);
                    const half *tile_ptr = &shmem[shmem_idx_a][k_step * WMMA_K];
                    wmma::load_matrix_sync(a[i], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);

                    #pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++) {
                        if (i == 0) {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * WMMA_N) * (warpId % 2) + (j * WMMA_N);
                            const half *tile_ptr = &shmem[shmem_idx_b][k_step * WMMA_K];
                            wmma::load_matrix_sync(b[j], tile_ptr, WMMA_K * CHUNK_K + SKEW_HALF);
                        }
                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // Store the D fragments to shared memory.
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++) {
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                #pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++) c[i][j].x[t] *= alpha;

                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * WMMA_K + j * WMMA_N;
                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }

        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global
        // memory.
        float *dst_gmem_warp_stream_ptr = &C[gmem_idx];

        #pragma unroll
        for (int i = 0; i < 16; i++) {
            *((int2 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int2 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        __syncthreads();
    }
}


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

int MAX_M_GLOBAL = 12544;
int MAX_N_GLOBAL = 2048;
int MAX_K_GLOBAL = 4608;
int MAX_COL_BUFFER = 802816;
int MAX_BOTTOM = 802816;
bool im2col_malloced = false;
bool gemm_malloced = false;
float *bottom;
float *col_buffer;
float *ori_host_A;
float *ori_host_B;
half *ori_wmma_A;
half *ori_wmma_B;
float *ori_wmma_C;

int hook_times = 0;

// hook cudnnConvolutionForward
cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
    // printf("cudnnConvolutionForward hooked!\n");
    // char foo;
    // std::cin >> foo;
    hook_times += 1;
    printf("hooked %d times\n", hook_times);
    cudaEvent_t startKERNEL, stopKERNEL;
	cudaErrCheck(cudaEventCreate(&startKERNEL));
	cudaErrCheck(cudaEventCreate(&stopKERNEL));
    float milliseconds = 0;

    cudaErrCheck(cudaEventRecord(startKERNEL));

    // img2col参数
    int input_n;
	int input_c;
	int input_h;
	int input_w;
	int output_n;
	int output_c;
	int output_h;
	int output_w;

	int col_n;
	int col_c;
	int col_h;
	int col_w;

    int kernel_k;
    int kernel_c;
    int kernel_h;
	int kernel_w;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
    int dilation_h;
	int dilation_w;

    // get input tensor args from xDesc
    cudnnDataType_t dataType;

    int a,b,c,d;

    CUDNN_SAFE_CALL(cudnnGetTensor4dDescriptor(xDesc, &dataType, &input_n, &input_c, &input_h, &input_w, &a, &b, &c, &d));
    // printf("input_n: %d, input_c: %d, input_h: %d, input_w: %d\n", input_n, input_c, input_h, input_w);
    // get Filter args from wDesc
    cudnnTensorFormat_t format;
    CUDNN_SAFE_CALL(cudnnGetFilter4dDescriptor(wDesc, &dataType, &format, &kernel_k, &kernel_c, &kernel_h, &kernel_w));
    // printf("kernel_k: %d, kernel_c: %d, kernel_h: %d, kernel_w: %d\n", kernel_k, kernel_c, kernel_h, kernel_w);

    // get output tensor args from yDesc
    CUDNN_SAFE_CALL(cudnnGetTensor4dDescriptor(yDesc, &dataType, &output_n, &output_c, &output_h, &output_w, &a, &b, &c, &d));
    // printf("output_n: %d, output_c: %d, output_h: %d, output_w: %d\n", output_n, output_c, output_h, output_w);

    cudnnConvolutionMode_t mode;
    // get convolution args from convDesc
    CUDNN_SAFE_CALL(cudnnGetConvolution2dDescriptor(convDesc, &pad_h, &pad_w, &stride_h, &stride_w, &dilation_h, &dilation_w, &mode, &dataType));
    // printf("pad_h: %d, pad_w: %d, stride_h: %d, stride_w: %d, dilation_h: %d, dilation_w: %d\n", pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);

    col_n = input_n;
    col_c = input_c;

    int height_col = (input_h + 2 * pad_h -
		(dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	int width_col = (input_w + 2 * pad_w -
		(dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	int num_kernels = input_n * input_c * height_col * width_col;

    assert(input_n == output_n);
    assert(output_h == height_col);
    assert(output_w == width_col);

    col_h = height_col;
    col_w = width_col;

    // printf("col_n:%d, col_c:%d, col_h:%d, col_w:%d\n", col_n, col_c, col_h, col_w);

    // std::cin >> foo;

    MAX_COL_BUFFER = max(MAX_COL_BUFFER, col_n * col_c * col_h * col_w);
    MAX_BOTTOM = max(MAX_BOTTOM, input_n * input_c * input_h * input_w);

    printf("MAX_COL_BUFFER: %d, MAX_BOTTOM: %d\n", MAX_COL_BUFFER, MAX_BOTTOM);
    
    if (!im2col_malloced) {
        cudaErrCheck(cudaMalloc((void**)&bottom, MAX_BOTTOM * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&col_buffer, MAX_COL_BUFFER * sizeof(float)));
        im2col_malloced = true;
    }

    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, bottom, input_n * input_c * input_h * input_w));
    curandErrCheck(curandGenerateUniform(gen, col_buffer, col_n * col_c * col_h * col_w));
    // cudaErrCheck(cudaMemset(bottom, 1.0f, input_n * input_c * input_h * input_w * sizeof(float)));
    // cudaErrCheck(cudaMemset(col_buffer, 1.0f, col_n * col_c * col_h * col_w * sizeof(float)));

    // 调用 img2col
    dim3 im_grid;
	dim3 im_block;
    im_block.x = 256;
	im_grid.x = int(num_kernels / 256);
	im_grid.x = 68 * 1;

    // printf("Running with im2col...\n");
    // printf("num_kernels: %d\n", num_kernels);
	// printf("input_h: %d\n", input_h);
	// printf("input_w: %d\n", input_w);
	// printf("kernel_h: %d\n", kernel_h);
	// printf("kernel_w: %d\n", kernel_w);
	// printf("pad_h: %d\n", pad_h);
	// printf("pad_w: %d\n", pad_w);
	// printf("stride_h: %d\n", stride_h);
	// printf("stride_w: %d\n", stride_w);
	// printf("dilation_h: %d\n", dilation_h);
	// printf("dilation_w: %d\n", dilation_w);
	// printf("height_col: %d\n", height_col);
	// printf("width_col: %d\n", width_col);
    // printf("col_buffer size: %d\n", col_n * col_c * col_h * col_w);
	// printf("bottom size: %d\n", input_n * input_c * input_h * input_w);

    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    printf("pre im2col_gpu_kernel took %f ms\n", milliseconds);
    milliseconds = 0;

    cudaErrCheck(cudaEventRecord(startKERNEL));
    // launch im2col
    checkKernelErrors((im2col_gpu_kernel<<<im_grid, im_block>>>(
		num_kernels, bottom, input_h, input_w, kernel_h, kernel_w, pad_h,
		pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
		width_col, col_buffer, input_n * input_c * input_h * input_w, col_n * col_c * col_h * col_w)));
    
    cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
	printf("im2col_gpu_kernel took %f ms\n", milliseconds);
    milliseconds = 0;


    // 调用 gemm
    // input matrix size
    // int M_INPUT = input_n * col_h; // 1 * 112
    // int N_INPUT = width_col;
    // int K_INPUT = kernel_k * kernel_c; // 64 * 3

    int M_INPUT = input_n * height_col * width_col; // 1 * 112 * 112
    int N_INPUT = kernel_k; // 64
    int K_INPUT = kernel_h * kernel_w * input_c;  // 7 * 7 * 3

    assert (M_INPUT * N_INPUT == output_n * output_c * output_h * output_w);



    int M_GLOBAL = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
	int N_GLOBAL = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
	int K_GLOBAL = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;

    MAX_M_GLOBAL = max(MAX_M_GLOBAL, M_GLOBAL);
    MAX_N_GLOBAL = max(MAX_N_GLOBAL, N_GLOBAL);
    MAX_K_GLOBAL = max(MAX_K_GLOBAL, K_GLOBAL);

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = 68 * 1;
	wmma_block.x = THREADS_PER_BLOCK;

    // half *ori_wmma_A = NULL;
	// half *ori_wmma_B = NULL;
	// float *ori_wmma_C = NULL;
    // float *ori_host_A = NULL;
	// float *ori_host_B = NULL;

    cudaErrCheck(cudaEventRecord(startKERNEL));

    if (!gemm_malloced) {
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * MAX_M_GLOBAL * MAX_K_GLOBAL));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * MAX_N_GLOBAL * MAX_K_GLOBAL));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), sizeof(half) * MAX_M_GLOBAL * MAX_K_GLOBAL));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), sizeof(half) * MAX_N_GLOBAL * MAX_K_GLOBAL));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), sizeof(float) * MAX_M_GLOBAL * MAX_N_GLOBAL));
        gemm_malloced = true;
    }

    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    printf("tzgemm malloc took %f ms\n", milliseconds);
    milliseconds = 0;

    assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	assert(((unsigned long long)ori_wmma_C) % 128 == 0);

    cudaErrCheck(cudaEventRecord(startKERNEL));

    // cudaErrCheck(cudaEventRecord(startKERNEL))
	curandErrCheck(curandGenerateUniform(gen, ori_host_A, M_GLOBAL * K_GLOBAL));
    curandErrCheck(curandGenerateUniform(gen, ori_host_B, N_GLOBAL * K_GLOBAL));
	convertFp32ToFp16 <<< (M_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL);
    convertFp32ToFp16 <<< (N_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL);
    cudaErrCheck(cudaMemset(ori_wmma_C, 0.0f, sizeof(float) * MAX_M_GLOBAL * MAX_N_GLOBAL));

    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    printf("tzgemm randgen & convert took %f ms\n", milliseconds);
    milliseconds = 0;

    // printf("Running with gemm...\n");
    // printf("gemm M_GLOBAL: %d, N_GLOBAL: %d, K_GLOBAL: %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    // printf("gemm block dim: %d, grid dim: %d\n", wmma_block_dim_x, wmma_grid_dim_x);
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C, 
		M_GLOBAL, N_GLOBAL, K_GLOBAL,
		// alpha, beta,
		wmma_grid_dim_x, wmma_block_dim_x)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    printf("ptb_tzgemm took %f ms\n", milliseconds);
    milliseconds = 0;

    // 调用cudnn测时
    cudnnStatus_t (*lcudnnConvolutionForward) (cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t, const void *, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *, size_t, const void *, const cudnnTensorDescriptor_t, void *) = (cudnnStatus_t (*) (cudnnHandle_t, const void *, const cudnnTensorDescriptor_t, const void *, const cudnnFilterDescriptor_t, const void *, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void *, size_t, const void *, const cudnnTensorDescriptor_t, void *))dlsym(RTLD_NEXT, "cudnnConvolutionForward");
    
    cudaErrCheck(cudaEventRecord(startKERNEL));
    lcudnnConvolutionForward(handle, alpha, xDesc, bottom, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    printf("cudnnConvolutionForward took %f ms\n", milliseconds);

    printf("M_ORI: %5d M_GLOBAL: %5d (%d x %d) \n", M_INPUT, M_GLOBAL, WMMA_M, M_TILES);
	printf("N_ORI: %5d N_GLOBAL: %5d (%d x %d) \n", N_INPUT, N_GLOBAL, WMMA_N, N_TILES);
	printf("K_ORI: %5d K_GLOBAL: %5d (%d x %d) \n", K_INPUT, K_GLOBAL, WMMA_K, K_TILES);

    printf("MAX_M_GLOBAL: %d, MAX_N_GLOBAL: %d, MAX_K_GLOBAL: %d\n", MAX_M_GLOBAL, MAX_N_GLOBAL, MAX_K_GLOBAL);

    printf("--------------------------------\n");

    return CUDNN_STATUS_SUCCESS;
}
