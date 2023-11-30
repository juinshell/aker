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


#include <mma.h>
using namespace nvcuda; // ??? 여기서 에러가 발생합니다. 

#include "header/stencil_header.h"
#include "file_t/stencil_kernel.cu"


int main(int argc, char* argv[]) {
    int stencil_blks = 3;
    int stencil_iter = 1;
    if (argc == 3) {
        stencil_blks = atoi(argv[1]);
        stencil_iter = atoi(argv[2]);
    } 

    // variables
    // ---------------------------------------------------------------------------------------
        float kernel_time;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    // stencil variables
    // ---------------------------------------------------------------------------------------
        float *host_stencil_ori_a0;
        float *stencil_ori_a0;
        float *stencil_ori_anext;

        float *host_stencil_ptb_a0;
        float *stencil_ptb_a0;
        float *stencil_ptb_anext;

        float c0=1.0f/6.0f;
        float c1=1.0f/6.0f/6.0f;

        // nx = 128 ny = 128 nz = 32 iter = 100
        // nx = 512 ny = 512 nz = 64 iter = 100
        int nx = 128 * 4;
        int ny = 128 * 4;
        int nz = 32 * 2;
        // int nz = 16 * 1;
        
        // printf("nx: %d, ny: %d, nz: %d, iteration: %d \n", nx, ny, nz, iteration);
        host_stencil_ori_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_ori_anext, nx * ny * nz * sizeof(float)));

        host_stencil_ptb_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
        cudaErrCheck(cudaMalloc((void**)&stencil_ptb_a0, nx * ny * nz * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&stencil_ptb_anext, nx * ny * nz * sizeof(float)));

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

        curandErrCheck(curandGenerateUniform(gen, stencil_ori_a0, nx * ny * nz));
        cudaErrCheck(cudaMemcpy(stencil_ori_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_ptb_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(stencil_ptb_anext, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
    // ---------------------------------------------------------------------------------------


    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 stencil_grid;
        dim3 stencil_block;
        stencil_block.x = tile_x;
        stencil_block.y = tile_y;
        stencil_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
        stencil_grid.y = (ny + tile_y - 1) / tile_y;

        printf("[ORI] Running with stencil...\n");
        printf("[ORI] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \n", 
            stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors(
                (ori_stencil<<<stencil_grid, stencil_block>>>(c0, c1, stencil_ori_a0, stencil_ori_anext, nx, ny, nz, stencil_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] stencil took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        int stencil_grid_dim_x = stencil_grid.x;
        int stencil_grid_dim_y = stencil_grid.y;
        // int stencil_block_dim_x = stencil_block.x;
        // int stencil_block_dim_y = stencil_block.y;
        stencil_grid.x = stencil_blks == 0 ? stencil_grid_dim_x * stencil_grid_dim_y : SM_NUM * stencil_blks;
        stencil_grid.y = 1;
        // stencil_block.x = stencil_block_dim_x * stencil_block_dim_y;
        // stencil_block.y = 1;
        
        printf("[PTB] Running with stencil...\n");
        printf("[PTB] stencil_grid -- %d * %d * %d stencil_block -- %d * %d * %d \n", 
            stencil_grid.x, stencil_grid.y, stencil_grid.z, stencil_block.x, stencil_block.y, stencil_block.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors(
                (ptb2_stencil<<<stencil_grid, stencil_block>>>(c0, c1, stencil_ptb_a0, stencil_ptb_anext, nx, ny, nz,
                    stencil_grid_dim_x, stencil_grid_dim_y, 
                    // stencil_block_dim_x, stencil_block_dim_y, 
                    stencil_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] stencil took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
        printf("Checking results...\n");
        cudaErrCheck(cudaMemcpy(host_stencil_ori_a0, stencil_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_stencil_ptb_a0, stencil_ptb_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int i = 0; i < nx * ny * nz; i++) {
            float v1 = host_stencil_ori_a0[i];
            float v2 = host_stencil_ptb_a0[i];
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

    cudaErrCheck(cudaFree(stencil_ori_a0));
    cudaErrCheck(cudaFree(stencil_ori_anext));
    cudaErrCheck(cudaFree(stencil_ptb_a0));
    cudaErrCheck(cudaFree(stencil_ptb_anext));

    free(host_stencil_ori_a0);
    free(host_stencil_ptb_a0);

    cudaErrCheck(cudaDeviceReset());
    return 0;

}