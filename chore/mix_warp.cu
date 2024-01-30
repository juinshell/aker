#include <stdio.h>
#include <curand.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

#define checkKernelErrors(expr)                                   \
    do                                                            \
    {                                                             \
        expr;                                                     \
                                                                  \
        cudaError_t __err = cudaGetLastError();                   \
        if (__err != cudaSuccess)                                 \
        {                                                         \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
                   cudaGetErrorString(__err));                    \
            abort();                                              \
        }                                                         \
    } while (0)

#include <cuda.h>
#include <mma.h>
using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 128
#define MATRIX_N 32 * 17
#define MATRIX_K 1024

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define NORMAL_M 256
#define NORMAL_N 256
#define NORMAL_K 256

#define TILE_N 16
#define TILE_TB_HEIGHT 8
#define TILE_M (TILE_N * TILE_TB_HEIGHT)

const float alpha = 2.0f;
const float beta = 2.0f;


__global__ void ori_wmma(half *a, half *b, float *c, int iteration) {
   // Leading dimensions. Packed with no transpositions.
    int lda = MATRIX_M;
    int ldb = MATRIX_K;
    int ldc = MATRIX_M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int loop = 0; loop < iteration; loop++) 
    {
        // for (int i = 0; i < 1; i++)
        for (int i = 0; i < MATRIX_K; i += WMMA_K)
        {
            int aRow = warpM * WMMA_M;
            int aCol = i;
            int bRow = i;
            int bCol = warpN * WMMA_N;

            // Bounds checking
            if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N)
            {
                // Load the inputs
                // ptx asm("load. bypass_l1cache")
                wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
                wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
                
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;

        if (cRow < MATRIX_M && cCol < MATRIX_N) {
            // for (int j = 0; j < 10000; j++) {
                wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
                for(int i=0; i < c_frag.num_elements; i++) {
                    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                }
                wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
            // }
        }
    }
}


__global__ void fp_compute(float *a, float *b, float *c)
{
    float a2 = 0,a3 = 1, a4 = 2, b2 = 2,b3 = 4,b4 = 5, c2 = 4,c3 = 7,c4 = 9;
    for (int j = 0; j < 4000; j++) {
        for (int t = 0; t < 30000; t++)
        {
            c2 += a2 * b2 + c2;
            c3 += a3 * b3 + c3;
            c4 += a4 * b4 + c4;
        }
    }

    c[0] += c2;
    c[1] += c3;
    c[2] += c4;
}


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}


__device__ void mix_wmma(half *a, half *b, float *c, int iteration) {  // Kt
   // Leading dimensions. Packed with no transpositions. 
    int lda = MATRIX_M;
    int ldb = MATRIX_K;
    int ldc = MATRIX_M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int loop = 0; loop < iteration; loop++) 
    {
        // for (int i = 0; i < 1; i++)
        for (int i = 0; i < MATRIX_K; i += WMMA_K)
        {
            int aRow = warpM * WMMA_M;
            int aCol = i;
            int bRow = i;
            int bCol = warpN * WMMA_N;

            // Bounds checking
            if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N)
            {
                // Load the inputs
                // ptx asm("load. bypass_l1cache")
                wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
                wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
                
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;

        if (cRow < MATRIX_M && cCol < MATRIX_N) {
            // for (int j = 0; j < 10000; j++) {
                wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
                for(int i=0; i < c_frag.num_elements; i++) {
                    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                }
                wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
            // }
        }
    }
}


__device__ void mix_compute(float *a, float *b, float *c) // Kc
{
    float a2 = 0,a3 = 1, a4 = 2, b2 = 2,b3 = 4,b4 = 5, c2 = 4,c3 = 7,c4 = 9;
    for (int j = 0; j < 4000; j++) {
        for (int t = 0; t < 30000; t++)
        {
            c2 += a2 * b2 + c2;
            c3 += a3 * b3 + c3;
            c4 += a4 * b4 + c4;
        }
    }

    c[0] += c2;
    c[1] += c3;
    c[2] += c4;
}


__global__ void mixwarp_kernel(
    half *wmma_a, half *wmma_b, float *wmma_c, int wmma_iter,
    float *fp_a, float *fp_b, float *fp_c
    ) {
    if (threadIdx.x < 128) {
        mix_wmma(wmma_a, wmma_b, wmma_c, wmma_iter);
    } else {
        mix_compute(fp_a, fp_b, fp_c);
    }
}


int main(int argc, char* argv[]) {
    float kernel_time;
    curandGenerator_t gen;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));

    cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaErrCheck(cudaStreamCreate(&streams[i]));
    }

	// wmma variables
    // ----------------------------------------------------------------------------------------------------------------------
    float *wmma_base_a;
    float *wmma_base_b;
    float *wmma_base_c;

    half *wmma_ori_a;
    half *wmma_ori_b;
    float *wmma_ori_c;
    half *wmma_mix_a;
    half *wmma_mix_b;
    float *wmma_mix_c;

    float *wmma_host_ori_c;
    float *wmma_host_mix_c;

    cudaErrCheck(cudaMalloc((void**)&wmma_base_a, MATRIX_M * MATRIX_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&wmma_base_b, MATRIX_K * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&wmma_base_c, MATRIX_M * MATRIX_N * sizeof(float)));

    cudaErrCheck(cudaMalloc((void**)&wmma_ori_a, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&wmma_ori_b, MATRIX_K * MATRIX_N * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&wmma_mix_a, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&wmma_mix_b, MATRIX_K * MATRIX_N * sizeof(half)));
    cudaErrCheck(cudaMalloc((void**)&wmma_mix_c, MATRIX_M * MATRIX_N * sizeof(float)));

    wmma_host_ori_c = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
    wmma_host_mix_c = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, wmma_base_a, MATRIX_M * MATRIX_K));
    curandErrCheck(curandGenerateUniform(gen, wmma_base_b, MATRIX_K * MATRIX_N));
    curandErrCheck(curandGenerateUniform(gen, wmma_base_c, MATRIX_M * MATRIX_N));

    convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_a, wmma_base_a, MATRIX_M * MATRIX_K);
    convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (wmma_ori_b, wmma_base_b, MATRIX_K * MATRIX_N);

    cudaErrCheck(cudaMemcpy(wmma_mix_a, wmma_ori_a, MATRIX_M * MATRIX_K * sizeof(half), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(wmma_mix_b, wmma_ori_b, MATRIX_K * MATRIX_N * sizeof(half), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(wmma_ori_c, wmma_base_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(wmma_mix_c, wmma_base_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

    // compute variables
    // ----------------------------------------------------------------------------------------------------------------------
    float *fp_ori_a;
    float *fp_ori_b;
    float *fp_ori_c;
    float *fp_mix_a;
    float *fp_mix_b;
    float *fp_mix_c;

    cudaErrCheck(cudaMalloc((void **)&fp_ori_a, 1 * NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&fp_ori_b, 1 * NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&fp_ori_c, 1 * NORMAL_M * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&fp_mix_a, 1 * NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&fp_mix_b, 1 * NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&fp_mix_c, 1 * NORMAL_M * NORMAL_N * sizeof(float)));

    int *ic_ori_a;
    int *ic_ori_b;
    int *ic_ori_c;
    int *ic_mix_a;
    int *ic_mix_b;
    int *ic_mix_c;

    cudaErrCheck(cudaMalloc((void **)&ic_ori_a, 1 * NORMAL_M * NORMAL_K * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&ic_ori_b, 1 * NORMAL_K * NORMAL_N * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&ic_ori_c, 1 * NORMAL_M * NORMAL_N * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&ic_mix_a, 1 * NORMAL_M * NORMAL_K * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&ic_mix_b, 1 * NORMAL_K * NORMAL_N * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&ic_mix_c, 1 * NORMAL_M * NORMAL_N * sizeof(int)));
    
    // First: using WMMA
    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 1;

    gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    // gridDim.x = 1;
    // gridDim.y = 68;

    printf("gridDim -- x y: %d, %d \n", gridDim.x, gridDim.y);
    printf("blockDim -- x y: %d, %d \n", blockDim.x, blockDim.y);

    printf("[ORI] Running with WMMA...\n");
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ori_wmma<<<gridDim, blockDim>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 40000)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] wmma took %f ms\n", kernel_time);

    printf("[ORI] Running with FP compute ...\n");
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((fp_compute<<<gridDim, blockDim>>>(fp_ori_a, fp_ori_b, fp_ori_c)));
    // checkKernelErrors((int_compute<<<gridDim, blockDim>>>(ic_ori_a, ic_ori_b, ic_ori_c)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] FP took %f ms\n", kernel_time);

    // blockDim.x = 128 * 2;

    printf("[MIX] Running with MIX WMMA FP32 ...\n");
    cudaErrCheck(cudaEventRecord(startKERNEL));
    // checkKernelErrors((ori_wmma<<<gridDim, blockDim, 0, streams[0]>>>(wmma_mix_a, wmma_mix_b, wmma_mix_c, 40000)));
    // // checkKernelErrors((fp_compute<<<gridDim, blockDim, 0, streams[0]>>>(fp_mix_a, fp_mix_a, wmma_mix_c)));
    checkKernelErrors((fp_compute<<<gridDim, blockDim, 0, streams[1]>>>(fp_mix_a, fp_mix_b, fp_mix_c)));
    checkKernelErrors((fp_compute<<<gridDim, blockDim, 0, streams[0]>>>(fp_ori_a, fp_ori_b, fp_ori_c)));
    // // checkKernelErrors((int_compute<<<gridDim, blockDim, 0, streams[1]>>>(ic_ori_a, ic_ori_b, ic_ori_c)));

    // checkKernelErrors((mixwarp_kernel<<<gridDim, blockDim>>>(wmma_mix_a, wmma_mix_b, wmma_mix_c, 40000,
    //                     fp_mix_a, fp_mix_b, fp_mix_c
    //                     )));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("mix took %f ms\n", kernel_time);


    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
