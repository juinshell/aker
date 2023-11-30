#include <stdio.h>
#include "switch.h"
#include <cuda.h>
#include <curand.h>

#include "header/fft_header.h"

#include "file_t/fft_kernel.cu"

#include "cuda_ipc.h"
// Define some error checking macros.
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
	int fft_blks = 2;
	int fft_iter = 18000;
	if (argc == 3)
	{
		fft_blks = atoi(argv[1]);
		fft_iter = atoi(argv[2]);
	}

	// variables
	// ---------------------------------------------------------------------------------------
	float kernel_time;
	cudaEvent_t start_kernel;
	cudaEvent_t stop_kernel;
	cudaErrCheck(cudaEventCreate(&start_kernel));
	cudaErrCheck(cudaEventCreate(&stop_kernel));

	// fft variables
	// ---------------------------------------------------------------------------------------
	//8*1024*1024;
	int n_bytes = N * B * sizeof(float2);
	int nthreads = T;
	srand(54321);

	float *host_shared_source = (float *)malloc(n_bytes);
	float2 *source = (float2 *)malloc(n_bytes);
	float2 *host_fft_ori_result = (float2 *)malloc(n_bytes);
	float2 *host_fft_pers_result = (float2 *)malloc(n_bytes);

	for (int b = 0; b < B; b++)
	{
		for (int i = 0; i < N; i++)
		{
			source[b * N + i].x = (rand() / (float)RAND_MAX) * 2 - 1;
			source[b * N + i].y = (rand() / (float)RAND_MAX) * 2 - 1;
		}
	}

	// allocate device memory
	float2 *fft_ori_source;
	float *fft_ori_shared_source;
	cudaMalloc((void **)&fft_ori_shared_source, n_bytes);
	// copy host memory to device
	cudaMemcpy(fft_ori_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&fft_ori_source, n_bytes);
	// copy host memory to device
	cudaMemcpy(fft_ori_source, source, n_bytes, cudaMemcpyHostToDevice);

	float2 *fft_pers_source;
	float *fft_pers_shared_source;
	cudaMalloc((void **)&fft_pers_shared_source, n_bytes);
	// copy host memory to device
	cudaMemcpy(fft_pers_shared_source, host_shared_source, n_bytes, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&fft_pers_source, n_bytes);
	// copy host memory to device
	cudaMemcpy(fft_pers_source, source, n_bytes, cudaMemcpyHostToDevice);

	// SOLO running
	// ---------------------------------------------------------------------------------------
	dim3 fft_grid;
	dim3 fft_block;
	fft_grid.x = B;
	fft_block.x = nthreads;
	printf("[ORI] Running with fft...\n");
	printf("[ORI] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\n", fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

	/* Barrier */
	int fd = shm_open(MMAP_FILE, O_RDWR, 0666);
	// ftruncate(fd, sizeof(pthread_barrier_t));

	pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	// pthread_barrierattr_t barrier_attr;
	// pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED);
	// pthread_barrier_init(shared_barrier, &barrier_attr, 2);

	pthread_barrier_wait(shared_barrier);

	cudaErrCheck(cudaEventRecord(start_kernel));
	ori_fft<<<fft_grid, fft_block>>>(fft_ori_source, fft_iter);
	cudaErrCheck(cudaEventRecord(stop_kernel));
	cudaErrCheck(cudaEventSynchronize(stop_kernel));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
	printf("[ORI] fft took %f ms\n\n", kernel_time);

	// PTB running
	// ---------------------------------------------------------------------------------------
	int fft_grid_dim_x = fft_grid.x;
	int fft_block_dim_x = fft_block.x;
	fft_grid.x = fft_blks == 0 ? fft_grid_dim_x : 68 * fft_blks;
	fft_block.x = fft_block_dim_x;
	printf("[PERS] Running with fft...\n");
	printf("[PERS] fft_grid -- %d * %d * %d fft_block -- %d * %d * %d\n", fft_grid.x, fft_grid.y, fft_grid.z, fft_block.x, fft_block.y, fft_block.z);

	pthread_barrier_wait(shared_barrier);
	cudaErrCheck(cudaEventRecord(start_kernel));
	pers_fft<<<fft_grid, fft_block>>>(fft_pers_source, fft_grid_dim_x, fft_block_dim_x, fft_iter);
	cudaErrCheck(cudaEventRecord(stop_kernel));
	cudaErrCheck(cudaEventSynchronize(stop_kernel));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
	printf("[PERS] fft took %f ms\n\n", kernel_time);

	// cudaErrCheck(cudaEventRecord(start_kernel));
	// pers_fft<<<fft_grid, fft_block>>>(fft_pers_source, fft_grid_dim_x, fft_block_dim_x, fft_iter);
	// cudaErrCheck(cudaEventRecord(stop_kernel));
	// cudaErrCheck(cudaEventSynchronize(stop_kernel));
	// cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
	// printf("[PERS] fft took %f ms\n\n", kernel_time);

	cudaErrCheck(cudaEventDestroy(start_kernel));
	cudaErrCheck(cudaEventDestroy(stop_kernel));
	cudaFree(fft_ori_source);
	free(host_shared_source);
	free(source);
	free(host_fft_ori_result);
}
