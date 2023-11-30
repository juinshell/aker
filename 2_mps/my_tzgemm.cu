#include <assert.h>
#include <stdio.h>
#include "switch.h"
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include "cuda_ipc.h"
using namespace nvcuda;

#include "header/tzgemm_header.h"

#include "file_t/tzgemm_kernel.cu"
#define curandErrCheck(stat)                     \
	{                                              \
		curandErrCheck_((stat), __FILE__, __LINE__); \
	}
void curandErrCheck_(curandStatus_t stat, const char *file, int line)
{
	if (stat != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
	}
}

#define cudaErrCheck(stat)                     \
	{                                            \
		cudaErrCheck_((stat), __FILE__, __LINE__); \
	}
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
	if (stat != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
	}
}

#define cublasErrCheck(stat)                     \
	{                                              \
		cublasErrCheck_((stat), __FILE__, __LINE__); \
	}
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
	}
}

#define checkKernelErrors(expr)                             \
	do                                                        \
	{                                                         \
		expr;                                                   \
                                                            \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess)                               \
		{                                                       \
			printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
						 cudaGetErrorString(__err));                    \
			abort();                                              \
		}                                                       \
	} while (0)

int main(int argc, char **argv)
{
	int tzgemm_blks = 2;
	int tzgemm_iter = 3400;
	int M_INPUT = 16 * 8 * 32;
	int N_INPUT = 16 * 8 * 24;
	int K_INPUT = 16 * 8 * 6;
	if (argc == 6)
	{
		tzgemm_blks = atoi(argv[1]);
		tzgemm_iter = atoi(argv[2]);
		M_INPUT = atoi(argv[3]);
		N_INPUT = atoi(argv[4]);
		K_INPUT = atoi(argv[5]);
	}

	cudaDeviceProp deviceProp;
	cudaErrCheck(cudaGetDeviceProperties(&deviceProp, 0));

	int M_GLOBAL = (M_INPUT < 64) ? 64 : (M_INPUT / 64) * 64;
	int N_GLOBAL = (N_INPUT < 64) ? 64 : (N_INPUT / 64) * 64;
	int K_GLOBAL = (K_INPUT < 64) ? 64 : (K_INPUT / 64) * 64;

	int M_TILES = M_GLOBAL / WMMA_M;
	int N_TILES = N_GLOBAL / WMMA_N;
	int K_TILES = K_GLOBAL / WMMA_K;

	printf("M_ORI: %5d M_GLOBAL: %5d (%d x %d) \n", M_INPUT, M_GLOBAL, WMMA_M, M_TILES);
	printf("N_ORI: %5d N_GLOBAL: %5d (%d x %d) \n", N_INPUT, N_GLOBAL, WMMA_N, N_TILES);
	printf("K_ORI: %5d K_GLOBAL: %5d (%d x %d) \n", K_INPUT, K_GLOBAL, WMMA_K, K_TILES);

	float *ori_host_A = NULL;
	float *ori_host_B = NULL;
	float *ori_result_C = NULL;

	half *ori_wmma_A = NULL;
	half *ori_wmma_B = NULL;
	float *ori_wmma_C = NULL;

	ori_result_C = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);

	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * N_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_A), sizeof(half) * M_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_B), sizeof(half) * N_GLOBAL * K_GLOBAL));
	cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_wmma_C), sizeof(float) * M_GLOBAL * N_GLOBAL));

	assert(((unsigned long long)ori_wmma_A) % 128 == 0);
	assert(((unsigned long long)ori_wmma_B) % 128 == 0);
	assert(((unsigned long long)ori_wmma_C) % 128 == 0);

	curandGenerator_t gen;
	curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
	curandErrCheck(curandGenerateUniform(gen, ori_host_A, M_GLOBAL * K_GLOBAL));
	curandErrCheck(curandGenerateUniform(gen, ori_host_B, N_GLOBAL * K_GLOBAL));
	convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(ori_wmma_A, ori_host_A, M_GLOBAL * K_GLOBAL);
	convertFp32ToFp16<<<(N_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(ori_wmma_B, ori_host_B, N_GLOBAL * K_GLOBAL);
	cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

	float milliseconds = 0;
	cudaEvent_t start_kernel, stop_kernel;
	cudaErrCheck(cudaEventCreate(&start_kernel));
	cudaErrCheck(cudaEventCreate(&stop_kernel));

	// SOLO running
	// ---------------------------------------------------------------------------------------

	dim3 wmma_grid;
	dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;

	cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));
	checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C,
																														M_GLOBAL, N_GLOBAL, K_GLOBAL,
																														// alpha, beta,
																														wmma_grid_dim_x, wmma_block_dim_x, tzgemm_iter)));

	cudaErrCheck(cudaDeviceSynchronize());

	/* Barrier */
	int fd = shm_open(MMAP_FILE, O_CREAT | O_RDWR, 0666);

	// ftruncate(fd, sizeof(pthread_barrier_t));

	pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);

	// pthread_barrierattr_t barrier_attr;
	// pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED);
	// pthread_barrier_init(shared_barrier, &barrier_attr, 2);

	pthread_barrier_wait(shared_barrier);
	// printf("Running with tzgemm \n");
	cudaErrCheck(cudaEventRecord(start_kernel));
	checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C,
																														M_GLOBAL, N_GLOBAL, K_GLOBAL,
																														// alpha, beta,
																														wmma_grid_dim_x, wmma_block_dim_x, tzgemm_iter)));
	cudaErrCheck(cudaEventRecord(stop_kernel));
	cudaErrCheck(cudaEventSynchronize(stop_kernel));
	cudaErrCheck(cudaEventElapsedTime(&milliseconds, start_kernel, stop_kernel));
	printf("[ORI] tzgemm took %f ms\n", milliseconds);

	// PTB running
	// ---------------------------------------------------------------------------------------
	wmma_grid.x = 68 * tzgemm_blks;
	wmma_block.x = THREADS_PER_BLOCK;
	pthread_barrier_wait(shared_barrier);
	cudaErrCheck(cudaEventRecord(start_kernel));
	checkKernelErrors((pers_tzgemm<<<wmma_grid, wmma_block>>>(ori_wmma_A, ori_wmma_B, ori_wmma_C,
																														M_GLOBAL, N_GLOBAL, K_GLOBAL,
																														// alpha, beta,
																														wmma_grid_dim_x, wmma_block_dim_x, tzgemm_iter)));
	cudaErrCheck(cudaEventRecord(stop_kernel));
	cudaErrCheck(cudaEventSynchronize(stop_kernel));
	cudaErrCheck(cudaEventElapsedTime(&milliseconds, start_kernel, stop_kernel));
	printf("[PERS] tzgemm took %f ms\n", milliseconds);

	cudaErrCheck(cudaEventDestroy(start_kernel));
	cudaErrCheck(cudaEventDestroy(stop_kernel));

	munmap(shared_barrier, sizeof(pthread_barrier_t));

	free(ori_result_C);
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_A)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_B)));
	cudaErrCheck(cudaFree(reinterpret_cast<void *>(ori_wmma_C)));

	return 0;
}
