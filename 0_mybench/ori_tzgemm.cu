#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>


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

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
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

using namespace nvcuda;


#include "header/tzgemm_ori.h"
#include "file_t/tzgemm_ori.cu"


int main(int argc, char **argv) {
	int tzgemm_blks = 1;
    int tzgemm_iter = 1;
	int M_INPUT = 16 * 8 * 32;
	int N_INPUT = 16 * 8 * 24;
	int K_INPUT = 16 * 8 * 6;
    if (argc == 6) {
        tzgemm_blks = atoi(argv[1]);
        tzgemm_iter = atoi(argv[2]);
        M_INPUT = atoi(argv[3]);
        N_INPUT = atoi(argv[4]);
        K_INPUT = atoi(argv[5]);
    } 

	cudaDeviceProp deviceProp;
	cudaErrCheck(cudaGetDeviceProperties(&deviceProp, 0));

	int M_GLOBAL = (M_INPUT < 128) ? 128 : (M_INPUT / 128) * 128;
	int N_GLOBAL = (N_INPUT < 128) ? 128 : (N_INPUT / 128) * 128;
	int K_GLOBAL = (K_INPUT < 128) ? 128 : (K_INPUT / 128) * 128;

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

	printf("M_ORI: %5d M_GLOBAL: %5d (%d x %d) \n", M_INPUT, M_GLOBAL, WMMA_M, M_TILES);
	printf("N_ORI: %5d N_GLOBAL: %5d (%d x %d) \n", N_INPUT, N_GLOBAL, WMMA_N, N_TILES);
	printf("K_ORI: %5d K_GLOBAL: %5d (%d x %d) \n", K_INPUT, K_GLOBAL, WMMA_K, K_TILES);

	float *ori_host_A = NULL;
	float *ori_host_B = NULL;
	float *ori_result_C = NULL;
	float *cublas_result_C = NULL;

	half *ori_wmma_A = NULL;
	half *ori_wmma_B = NULL;
	float *ori_wmma_C = NULL;
	float *cublas_wmma_C = NULL;

	ori_result_C = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
	cublas_result_C = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * N_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), sizeof(half) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), sizeof(half) * N_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), sizeof(float) * M_GLOBAL * N_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cublas_wmma_C), sizeof(float) * M_GLOBAL * N_GLOBAL));

	assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	assert(((unsigned long long)ori_wmma_C) % 128 == 0);
	assert(((unsigned long long)cublas_wmma_C) % 128 == 0);

	curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
	curandErrCheck(curandGenerateUniform(gen, ori_host_A, M_GLOBAL * K_GLOBAL));
    curandErrCheck(curandGenerateUniform(gen, ori_host_B, N_GLOBAL * K_GLOBAL));
	checkKernelErrors((convertFp32ToFp16 <<< (M_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL)));
    checkKernelErrors((convertFp32ToFp16 <<< (N_GLOBAL * K_GLOBAL + 255) / 256, 256 >>> (ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL)));
	cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));
	cudaErrCheck(cudaMemset(cublas_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

	float milliseconds = 0;
	cudaEvent_t startKERNEL, stopKERNEL;
	cudaErrCheck(cudaEventCreate(&startKERNEL));
	cudaErrCheck(cudaEventCreate(&stopKERNEL));

	// SOLO running
    // ---------------------------------------------------------------------------------------
	dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	// wmma_grid.x = 68 * tzgemm_blks;
	// wmma_block.x = THREADS_PER_BLOCK;
	printf("[PTB] wmma_grid.x -- %d \n", wmma_grid.x);

	int SHMEM_SZ = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
	// SHMEM_SZ = sizeof(half) * (BLOCK_COL_TILES * WMMA_M) * (CHUNK_K * WMMA_K + SKEW_HALF) * 2;
	printf("SHMEM_SZ %d \n", SHMEM_SZ);
    cudaErrCheck(cudaFuncSetAttribute(
		ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C, 
		M_GLOBAL, N_GLOBAL, K_GLOBAL,
		// alpha, beta,
		wmma_grid_dim_x, wmma_block_dim_x, 1)));
	cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

	printf("Running with tzgemm \n");
	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ>>>(
							ori_wmma_A, ori_wmma_B, ori_wmma_C, 
							M_GLOBAL, N_GLOBAL, K_GLOBAL,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, tzgemm_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
	printf("tzgemm took %f ms\n", milliseconds);

	cublasHandle_t cublasHandle;
	cublasErrCheck(cublasCreate(&cublasHandle));
	cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
	printf("Running with cuBLAS...\n");
	cudaErrCheck(cudaEventRecord(startKERNEL));
	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
                        N_GLOBAL, M_GLOBAL, K_GLOBAL, 
                        &alpha_g,
                        ori_wmma_B, CUDA_R_16F, K_GLOBAL,
                        ori_wmma_A, CUDA_R_16F, K_GLOBAL,
                        &beta_g, 
                        cublas_wmma_C, CUDA_R_32F, N_GLOBAL,
                        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
	printf("cublas took %f ms\n\n", milliseconds);

	cudaErrCheck(cudaMemcpy(ori_result_C, ori_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
	cudaErrCheck(cudaMemcpy(cublas_result_C, cublas_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

	printf("Checking results...\n");
	int count = 0;
	for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++) {
		if (fabs(ori_result_C[i] - cublas_result_C[i]) > 0.1f) {
		count++;
		if (count < 10) printf("%f %f\n", ori_result_C[i], cublas_result_C[i]);
		}
		if(i < 3) printf("%f %f\n", ori_result_C[i], cublas_result_C[i]);
	}
	if (count > 0) {
		printf("[Errors] %d errors in %d numbers.\n", count, N_GLOBAL * M_GLOBAL);
	} else {
		printf("[Success]!!!\n");
	}

	free(ori_result_C);
	free(cublas_result_C);
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_A)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_B)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_C)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(cublas_wmma_C)));

	return 0;
}
