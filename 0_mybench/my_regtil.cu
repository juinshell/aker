#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

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

#include "header/regtil_header.h"
#include "file_t/regtil_kernel.cu"


int main(int argc, char* argv[]) {
    int regtil_blks = 1;
    int regtil_iter = 1;
    if (argc == 3) {
        regtil_blks = atoi(argv[1]);
        regtil_iter = atoi(argv[2]);
    } 

    // variables
    // ---------------------------------------------------------------------------------------
        float kernel_time;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    // regtil variables
    // ---------------------------------------------------------------------------------------
        float *host_regtil_ori_a0;
        float *regtil_ori_a0;
        float *regtil_ori_anext;

        float *host_regtil_ptb_a0;
        float *regtil_ptb_a0;
        float *regtil_ptb_anext;

        float c0=1.0f/6.0f;
        float c1=1.0f/6.0f/6.0f;

        // nx = 128 ny = 128 nz = 32 iter = 100
        // nx = 512 ny = 512 nz = 64 iter = 100
        int nx = 128 * 4;
        int ny = 128 * 4;
        int nz = 32 * 2;

        // printf("nx: %d, ny: %d, nz: %d, iteration: %d \n", nx, ny, nz, iteration);
        host_regtil_ori_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&regtil_ori_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&regtil_ori_anext, nx * ny * nz * sizeof(float)));

        host_regtil_ptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&regtil_ptb_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&regtil_ptb_anext, nx * ny * nz * sizeof(float)));

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

        curandErrCheck(curandGenerateUniform(gen, regtil_ori_a0, nx * ny * nz));
        cudaErrCheck(cudaMemcpy(regtil_ori_anext, regtil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(regtil_ptb_a0, regtil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(regtil_ptb_anext, regtil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
    // ---------------------------------------------------------------------------------------

    // ORI running
    // ---------------------------------------------------------------------------------------
        dim3 regtil_grid;
        dim3 regtil_block;
        regtil_block.x = tile_x;
        regtil_block.y = tile_y;
        regtil_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
        regtil_grid.y = (ny + tile_y - 1) / tile_y;

        printf("[ORI] Running with regtil...\n");
        printf("[ORI] regtil_grid -- %d * %d * %d regtil_block -- %d * %d * %d \n", 
            regtil_grid.x, regtil_grid.y, regtil_grid.z, regtil_block.x, regtil_block.y, regtil_block.z);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors(
                (ori_regtil<<<regtil_grid, regtil_block>>>(c0, c1, regtil_ori_a0, regtil_ori_anext, nx, ny, nz, regtil_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] regtil took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        int regtil_grid_dim_x = regtil_grid.x;
        int regtil_grid_dim_y = regtil_grid.y;
        int regtil_block_dim_x = regtil_block.x;
        int regtil_block_dim_y = regtil_block.y;
        regtil_grid.x = regtil_blks == 0 ? regtil_grid_dim_x * regtil_grid_dim_y : SM_NUM * regtil_blks;
        regtil_grid.y = 1;
        regtil_block.x = regtil_block_dim_x * regtil_block_dim_y;
        regtil_block.y = 1;

        printf("[PTB] Running with regtil...\n");
        printf("[PTB] regtil_grid -- %d * %d * %d regtil_block -- %d * %d * %d \n", 
            regtil_grid.x, regtil_grid.y, regtil_grid.z, regtil_block.x, regtil_block.y, regtil_block.z);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors(
                (ptb_regtil<<<regtil_grid, regtil_block>>>(c0, c1, regtil_ptb_a0, regtil_ptb_anext, nx, ny, nz,
                    regtil_grid_dim_x, regtil_grid_dim_y, regtil_block_dim_x, regtil_block_dim_y, regtil_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] regtil took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
        printf("Checking results...\n");
        cudaErrCheck(cudaMemcpy(host_regtil_ori_a0, regtil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_regtil_ptb_a0, regtil_ptb_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int i = 0; i < nx * ny * nz; i++) {
            float v1 = host_regtil_ori_a0[i];
            float v2 = host_regtil_ptb_a0[i];
            if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
        }
    // ---------------------------------------------------------------------------------------

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaFree(regtil_ori_a0));
    cudaErrCheck(cudaFree(regtil_ori_anext));
    cudaErrCheck(cudaFree(regtil_ptb_a0));
    cudaErrCheck(cudaFree(regtil_ptb_anext));

    free(host_regtil_ori_a0);
    free(host_regtil_ptb_a0);

    cudaErrCheck(cudaDeviceReset());
    return 0;

}