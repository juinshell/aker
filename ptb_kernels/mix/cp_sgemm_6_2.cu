
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include <malloc.h>
#include <sys/time.h>
using namespace nvcuda; 

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

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
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

#include "header/cp_header.h"
#include "kernel/cp_kernel.cu"

#include "header/sgemm_header.h"
#include "kernel/sgemm_kernel.cu"

#include "mix_kernel/cp-sgemm.cu" 

int main(int argc, char* argv[]) {
    int errors = 0;

	  // variables
    // ---------------------------------------------------------------------------------------
		float kernel_time;
		cudaEvent_t startKERNEL;
		cudaEvent_t stopKERNEL;
		cudaErrCheck(cudaEventCreate(&startKERNEL));
		cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    // cp variables
    // ---------------------------------------------------------------------------------------
		int cp_blks = 8;
	    int cp_iter = 1;
        float *atoms = NULL;
		int atomcount = ATOMCOUNT;
		const float gridspacing = 0.1;					// number of atoms to simulate
		dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
		initatoms(&atoms, atomcount, volsize, gridspacing);

		// allocate and initialize the GPU output array
		int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

        float *ori_output;	
		float *ptb_output;
		float *gptb_output;
		cudaErrCheck(cudaMalloc((void**)&ori_output, volmemsz));
		cudaErrCheck(cudaMemset(ori_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
		cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
		cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
		float *host_ori_energy = (float *) malloc(volmemsz);
		float *host_ptb_energy = (float *) malloc(volmemsz);
		float *host_gptb_energy = (float *) malloc(volmemsz);

        dim3 cp_grid, cp_block, ori_cp_grid, ori_cp_block;
        int atomstart = 1;
		int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
		cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
		cp_block.y = BLOCKSIZEY;
		cp_block.z = 1;
		cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
		cp_grid.y = volsize.y / cp_block.y; 
		cp_grid.z = volsize.z / cp_block.z; 
		ori_cp_grid = cp_grid;
		ori_cp_block = cp_block;
		printf("[ORI] Running with cp...\n");
		printf("[ORI] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ori_output, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] cp took %f ms\n\n", kernel_time);

        cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

	// PTB running
    // ---------------------------------------------------------------------------------------
        int solo_ptb_cp_blks = 6;
	    cp_iter = 1;
		int cp_grid_dim_x = cp_grid.x;
		int cp_grid_dim_y = cp_grid.y;
		cp_grid.x = solo_ptb_cp_blks == 0 ? cp_grid_dim_x * cp_grid_dim_y : SM_NUM * solo_ptb_cp_blks;
		cp_grid.y = 1;
		printf("[PTB] Running with cp...\n");
		printf("[PTB] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb2_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ptb_output, 
			cp_grid_dim_x, cp_grid_dim_y, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] cp took %f ms\n\n", kernel_time);

        cudaMemcpy(host_ptb_energy, ptb_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

    // sgemm variables
    // ---------------------------------------------------------------------------------------
        int sgemm_blks = 4;
        int sgemm_iter = 1;
        float *sgemm_ori_a;
        float *sgemm_ori_b;
        float *sgemm_ori_c;
        float *sgemm_ptb_a;
        float *sgemm_ptb_b;
        float *sgemm_ptb_c;
        float *sgemm_gptb_a;
        float *sgemm_gptb_b;
        float *sgemm_gptb_c;
        float *host_sgemm_ori_c;
        float *host_sgemm_ptb_c;
        float *host_sgemm_gptb_c;
        

        // parallel experiment
        int NORMAL_M = 4096;
        int NORMAL_N = 4128;
        int NORMAL_K = 4064;

        NORMAL_M = (NORMAL_M / 10) * sgemm_iter;

        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&sgemm_gptb_c, NORMAL_M * NORMAL_N * sizeof(float)));

        host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        host_sgemm_ptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
        host_sgemm_gptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));

        curandGenerator_t sgemm_gen;
        curandErrCheck(curandCreateGenerator(&sgemm_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(sgemm_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
        curandErrCheck(curandGenerateUniform(sgemm_gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
        cudaErrCheck(cudaMemcpy(sgemm_ptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_ptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_gptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(sgemm_gptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
        curandErrCheck(curandDestroyGenerator(sgemm_gen));
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 sgemm_grid, ori_sgemm_grid;
        dim3 sgemm_block, ori_sgemm_block;
        sgemm_block.x = TILE_N;
        sgemm_block.y = TILE_TB_HEIGHT;
        sgemm_grid.x = NORMAL_M/TILE_M;
        sgemm_grid.y = NORMAL_N/TILE_N;
        ori_sgemm_grid = sgemm_grid;
        ori_sgemm_block = sgemm_block;
        printf("[ORI] Running with sgemm...\n");
        printf("[ORI] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_sgemm <<< sgemm_grid, sgemm_block >>> (
                    sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
                    NORMAL_M, NORMAL_N, NORMAL_K, 1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] sgemm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        int sgemm_grid_dim_x = sgemm_grid.x;
        int sgemm_grid_dim_y = sgemm_grid.y;
        // int sgemm_block_dim_x = sgemm_block.x;
        // int sgemm_block_dim_y = sgemm_block.y;
        sgemm_grid.x = sgemm_grid_dim_x * sgemm_grid_dim_y;
        sgemm_grid.x = sgemm_blks == 0 ? sgemm_grid_dim_x * sgemm_grid_dim_y : 68 * sgemm_blks;
        sgemm_grid.y = 1;
        // sgemm_block.x = sgemm_block_dim_x * sgemm_block_dim_y;
        // sgemm_block.y = 1;
        printf("[PTB] Running with sgemm...\n");
        printf("[PTB] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
            sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb2_sgemm <<< sgemm_grid, sgemm_block >>> (
                    sgemm_ptb_a, sgemm_ptb_b, sgemm_ptb_c, 
                    NORMAL_M, NORMAL_N, NORMAL_K,
                    sgemm_grid_dim_x, sgemm_grid_dim_y, 
                    1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] sgemm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(1024, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_cp_sgemm_kernel_6_2 <<<mix_kernel_grid, mix_kernel_block>>>(runatoms, 0.1, gptb_output, ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z, 
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cp_grid.x * ori_cp_grid.y * ori_cp_grid.z, sgemm_gptb_a, sgemm_gptb_b, sgemm_gptb_c, NORMAL_M, NORMAL_N, NORMAL_K,
            ori_sgemm_grid.x, ori_sgemm_grid.y, ori_sgemm_grid.z, ori_sgemm_block.x, ori_sgemm_block.y, ori_sgemm_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cp_sgemm 6_2 took %f ms\n\n", kernel_time);
  // ---------------------------------------------------------------------------------------


	// Checking results
    // ---------------------------------------------------------------------------------------
    	cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
	    cudaMemcpy(host_gptb_energy, gptb_output, volmemsz,  cudaMemcpyDeviceToHost);
            
        errors = 0;
        for (int i = 0; i < volsize.x * volsize.y * volsize.z; i++) {
            float v1 = host_ori_energy[i];
            float v2 = host_gptb_energy[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 10) printf("%f %f\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORI VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIG VERSION and GPTB VERSION agree.\n");
        }
	// ---------------------------------------------------------------------------------------

    // Checking results
    // ---------------------------------------------------------------------------------------
        cudaErrCheck(cudaMemcpy(host_sgemm_ori_c, sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_sgemm_gptb_c, sgemm_gptb_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));

        errors = 0;
        for (int i = 0; i < NORMAL_M * NORMAL_N; i++) {
            float v1 = host_sgemm_ori_c[i];
            float v2 = host_sgemm_gptb_c[i];
            if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
        }
    // ---------------------------------------------------------------------------------------


}
