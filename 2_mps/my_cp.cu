/*
 * Author: raphael hao
 */

#include <stdio.h>
#include <stdlib.h>
#include "switch.h"
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include "header/cp_header.h"
#include "file_t/cp_kernel.cu"
#include "cuda_ipc.h"
#include <cuda.h>

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
	int cp_blks = 4;
	int cp_iter = 2700;
	if (argc == 3)
	{
		cp_blks = atoi(argv[1]);
		cp_iter = atoi(argv[2]);
	}

	// variables
	// ---------------------------------------------------------------------------------------
	float kernel_time;
	cudaEvent_t start_kernel;
	cudaEvent_t stop_kernel;
	cudaErrCheck(cudaEventCreate(&start_kernel));
	cudaErrCheck(cudaEventCreate(&stop_kernel));

	// cp variables
	// ---------------------------------------------------------------------------------------
	float *atoms = NULL;
	int atomcount = ATOMCOUNT;
	const float gridspacing = 0.1; // number of atoms to simulate
	dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
	initatoms(&atoms, atomcount, volsize, gridspacing);

	// allocate and initialize the GPU output array
	int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

	float *ori_output;
	float *pers_output;
	cudaErrCheck(cudaMalloc((void **)&ori_output, volmemsz));
	cudaErrCheck(cudaMemset(ori_output, 0, volmemsz));
	cudaErrCheck(cudaMalloc((void **)&pers_output, volmemsz));
	cudaErrCheck(cudaMemset(pers_output, 0, volmemsz));
	float *host_ori_energy = (float *)malloc(volmemsz);
	float *host_pers_energy = (float *)malloc(volmemsz);

	// SOLO running
	// ---------------------------------------------------------------------------------------
	dim3 cp_grid, cp_block;
	cp_block.x = BLOCKSIZEX; // each thread does multiple Xs
	cp_block.y = BLOCKSIZEY;
	cp_block.z = 1;
	cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
	cp_grid.y = volsize.y / cp_block.y;
	cp_grid.z = volsize.z / cp_block.z;
	printf("[ORI] Running with cp...\n");
	printf("[ORI] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n",
				 cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

	int atomstart = 1;
	int runatoms = MAXATOMS;
	copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0 * gridspacing);

	/* Barrier */
	int fd = shm_open(MMAP_FILE, O_RDWR, 0666);
	// ftruncate(fd, sizeof(pthread_barrier_t));

	pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	// pthread_barrierattr_t barrier_attr;
	// pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED);
	// pthread_barrier_init(shared_barrier, &barrier_attr, 2);

	pthread_barrier_wait(shared_barrier);
	cudaErrCheck(cudaEventRecord(start_kernel));
	checkKernelErrors((ori_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ori_output, cp_iter)));
	cudaErrCheck(cudaEventRecord(stop_kernel));
	cudaErrCheck(cudaEventSynchronize(stop_kernel));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
	printf("cp took %f ms\n\n", kernel_time);

	// PTB running
	// ---------------------------------------------------------------------------------------
	int cp_grid_dim_x = cp_grid.x;
	int cp_grid_dim_y = cp_grid.y;
	int cp_block_dim_x = cp_block.x;
	int cp_block_dim_y = cp_block.y;
	cp_grid.x = 68 * 4;
	cp_grid.x = cp_blks == 0 ? cp_grid_dim_x * cp_grid_dim_y : 80 * cp_blks;
	cp_grid.y = 1;
	cp_block.x = cp_block_dim_x * cp_block_dim_y;
	cp_block.y = 1;
	printf("[PERS] Running with cp...\n");
	printf("[PERS] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n",
				 cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

	atomstart = 1;
	runatoms = MAXATOMS;
	copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0 * gridspacing);

	pthread_barrier_wait(shared_barrier);
	cudaErrCheck(cudaEventRecord(start_kernel));
	checkKernelErrors((pers_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, pers_output,
																											 cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, cp_iter)));
	cudaErrCheck(cudaEventRecord(stop_kernel));
	cudaErrCheck(cudaEventSynchronize(stop_kernel));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
	printf("[PERS] cp took %f ms\n", kernel_time);


	//!
	//! \brief solo ptb run
	//!
	//!

	// cudaErrCheck(cudaEventRecord(start_kernel));
	// checkKernelErrors((pers_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, pers_output,
	// 																										 cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, cp_iter)));
	// cudaErrCheck(cudaEventRecord(stop_kernel));
	// cudaErrCheck(cudaEventSynchronize(stop_kernel));
	// cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
	// printf("[PERS] cp took %f ms\n", kernel_time);

	cudaErrCheck(cudaEventDestroy(start_kernel));
	cudaErrCheck(cudaEventDestroy(stop_kernel));

	cudaFree(ori_output);
	free(atoms);
	free(host_ori_energy);

	return 0;
}
