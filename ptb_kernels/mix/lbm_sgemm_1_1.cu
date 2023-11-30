
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

#include "header/lbm_header.h"
#include "kernel/lbm_kernel.cu"

#include "header/sgemm_header.h"
#include "kernel/sgemm_kernel.cu"

#include "mix_kernel/lbm-sgemm.cu" 

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

    // lbm variables
    // ---------------------------------------------------------------------------------------
        int lbm_blks = 1;
        int lbm_iter = 1;
        float *lbm_ori_src;
        float *lbm_ori_dst;
        float *lbm_ptb_src;
        float *lbm_ptb_dst;
        float *lbm_gptb_src;
        float *lbm_gptb_dst;
        float *host_lbm_ori_dst;
        float *host_lbm_ptb_dst;
        float *host_lbm_gptb_dst;

        const size_t size = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float) + 2 * TOTAL_MARGIN * sizeof(float);

        host_lbm_ori_dst = (float *)malloc(size);
        host_lbm_ptb_dst = (float *)malloc(size);
        host_lbm_gptb_dst = (float *)malloc(size);
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ori_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_ptb_dst, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_gptb_src, size));
        cudaErrCheck(cudaMalloc((void **)&lbm_gptb_dst, size));

        curandGenerator_t lbm_gen;
        curandErrCheck(curandCreateGenerator(&lbm_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(lbm_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_src, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        curandErrCheck(curandGenerateUniform(lbm_gen, lbm_ori_dst, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
        cudaErrCheck(cudaMemcpy(lbm_ptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_ptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_gptb_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(lbm_gptb_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
        lbm_ori_src += REAL_MARGIN;
        lbm_ori_dst += REAL_MARGIN;
        lbm_ptb_src += REAL_MARGIN;
        lbm_ptb_dst += REAL_MARGIN;
        lbm_gptb_src += REAL_MARGIN;
        lbm_gptb_dst += REAL_MARGIN;

    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 lbm_block, lbm_grid, ori_lbm_block, ori_lbm_grid;
        lbm_block.x = SIZE_X;
        lbm_grid.x = SIZE_Y;
        lbm_grid.y = SIZE_Z;
        lbm_block.y = lbm_block.z = lbm_grid.z = 1;
        ori_lbm_block = lbm_block;
        ori_lbm_grid = lbm_grid;
        printf("[ORI] Running with lbm...\n");
        printf("[ORI] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_lbm<<<lbm_grid, lbm_block>>>(lbm_ori_src, lbm_ori_dst, lbm_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] lbm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    // PTB running
    // ---------------------------------------------------------------------------------------
        int lbm_block_dim_x = lbm_block.x;
        int lbm_block_dim_y = lbm_block.y;
        int lbm_block_dim_z = lbm_block.z;
        int lbm_grid_dim_x = lbm_grid.x;
        int lbm_grid_dim_y = lbm_grid.y;
        int lbm_grid_dim_z = lbm_grid.z;

        lbm_grid.x = lbm_blks == 0 ? lbm_grid_dim_x * lbm_grid_dim_y : SM_NUM * lbm_blks;
        lbm_grid.y = lbm_grid.z = 1;
        lbm_block.x = lbm_block_dim_x * lbm_block_dim_y * lbm_block_dim_z;
        lbm_block.y = lbm_block.z = 1;
        printf("[PTB] Running with lbm...\n");
        printf("[PTB] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", 
            lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_lbm<<<lbm_grid, lbm_block>>>(lbm_ptb_src, lbm_ptb_dst,
            lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
            lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z, lbm_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] lbm took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


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
        dim3 mix_kernel_grid = dim3(272, 1, 1);
        dim3 mix_kernel_block = dim3(256, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_lbm_sgemm_kernel_1_1 <<<mix_kernel_grid, mix_kernel_block>>>(lbm_gptb_src, lbm_gptb_dst,
            ori_lbm_grid.x, ori_lbm_grid.y, ori_lbm_grid.z, ori_lbm_block.x, ori_lbm_block.y, ori_lbm_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_lbm_grid.x * ori_lbm_grid.y * ori_lbm_grid.z, sgemm_gptb_a, sgemm_gptb_b, sgemm_gptb_c, NORMAL_M, NORMAL_N, NORMAL_K,
            ori_sgemm_grid.x, ori_sgemm_grid.y, ori_sgemm_grid.z, ori_sgemm_block.x, ori_sgemm_block.y, ori_sgemm_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] lbm_sgemm 1_1 took %f ms\n\n", kernel_time);
  // ---------------------------------------------------------------------------------------


    // ---------------------------------------------------------------------------------------
        lbm_ori_src -= REAL_MARGIN;
        lbm_ori_dst -= REAL_MARGIN;
        lbm_ptb_src -= REAL_MARGIN;
        lbm_ptb_dst -= REAL_MARGIN;
        lbm_gptb_src -= REAL_MARGIN;
        lbm_gptb_dst -= REAL_MARGIN;
        cudaErrCheck(cudaMemcpy(host_lbm_ori_dst, lbm_ori_dst, size, cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_lbm_gptb_dst, lbm_gptb_dst, size, cudaMemcpyDeviceToHost));
        errors = 0;
        for (int i = 0; i < TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN; i++) {
            float v1 = host_lbm_ori_dst[i];
            float v2 = host_lbm_gptb_dst[i];
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
