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
using namespace nvcuda; 


#include "header/img_header.h"
#include "file_t/img_kernel.cu"


int main(int argc, char* argv[]) {
    int img_blks = 1;
    int img_iter = 1;
    if (argc == 3) {
        img_blks = atoi(argv[1]);
        img_iter = atoi(argv[2]);
    } 

    // variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));

    // img variables
    // ---------------------------------------------------------------------------------------
        int DATA_W = 1536;
        int DATA_H = 1024;
        int BINS = 256;

        unsigned int *host_data;
        unsigned int *img_ori_data;
        unsigned int *img_ori_result;
        unsigned int *host_img_ori_result;
        unsigned int *img_ptb_data;
        unsigned int *img_ptb_result;
        unsigned int *host_img_ptb_result;

        char inpFiles[] = "./file_t/img_input.iml";
        host_data = (unsigned int *)malloc(DATA_W * DATA_H * sizeof(unsigned int));
        readImage(inpFiles, host_data, DATA_W * DATA_H);

        cudaErrCheck(cudaMalloc((void**)&img_ori_data, DATA_W * DATA_H * 60 * sizeof(unsigned int)));
        cudaErrCheck(cudaMalloc((void**)&img_ori_result, BINS * sizeof(unsigned int)));
        cudaErrCheck(cudaMalloc((void**)&img_ptb_data, DATA_W * DATA_H * 60 * sizeof(unsigned int)));
        cudaErrCheck(cudaMalloc((void**)&img_ptb_result, BINS * sizeof(unsigned int)));
        cudaErrCheck(cudaMemset(img_ori_result, 0, BINS * sizeof(unsigned int)));
        cudaErrCheck(cudaMemset(img_ptb_result, 0, BINS * sizeof(unsigned int)));

        host_img_ori_result = (unsigned int *)malloc(BINS*sizeof(unsigned int));
        host_img_ptb_result = (unsigned int *)malloc(BINS*sizeof(unsigned int));

        for (int i = 0; i < 60; i++) {
            cudaErrCheck(cudaMemcpy(img_ptb_data + (DATA_W * DATA_H * i), host_data, DATA_W * DATA_H * sizeof(unsigned int), cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(img_ori_data + (DATA_W * DATA_H * i), host_data, DATA_W * DATA_H * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }

        DATA_W = DATA_W * 6 * img_iter;
        
    // ---------------------------------------------------------------------------------------


    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 img_grid;
        dim3 img_block;
        img_grid.x = NUM_BLOCKS;
        img_block.x = THREADS;

        printf("[ORI] Running with img...\n");
        printf("[ORI] img_grid -- %d * %d * %d img_block -- %d * %d * %d \n", 
            img_grid.x, img_grid.y, img_grid.z, img_block.x, img_block.y, img_block.z);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_img<<<img_grid, img_block>>>(img_ori_result, img_ori_data, DATA_H * DATA_W, BINS, 1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] img took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        int grid_dimension_x = img_grid.x;
        int block_dimension_x = img_block.x;
        img_grid.x = img_blks == 0 ? grid_dimension_x : SM_NUM * img_blks;;

        printf("[PTB] Running with img...\n");
        printf("[PTB] img_grid -- %d * %d * %d img_block -- %d * %d * %d \n", 
            img_grid.x, img_grid.y, img_grid.z, img_block.x, img_block.y, img_block.z);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_img<<<img_grid, img_block>>>(img_ptb_result, img_ptb_data, DATA_H * DATA_W, BINS, 
                        grid_dimension_x, block_dimension_x, 1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] img took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // Error checking
    // ---------------------------------------------------------------------------------------
        printf("Checking results...\n");
        cudaErrCheck(cudaMemcpy(host_img_ori_result, img_ori_result, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_img_ptb_result, img_ptb_result, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int i = 0; i < BINS; i++) {
            unsigned int v1 = host_img_ori_result[i];
            unsigned int v2 = host_img_ptb_result[i];
            if (v1 - v2 != 0) {
                errors++;
                if (errors < 10) printf("%u %u \n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("[IMG] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
        } else {
            printf("[IMG] Results verified: ORIGIN VERSION and MY VERSION agree.\n\n");
        }
    // ---------------------------------------------------------------------------------------

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaFree(img_ori_data));
    cudaErrCheck(cudaFree(img_ori_result));
    cudaErrCheck(cudaFree(img_ptb_data));
    cudaErrCheck(cudaFree(img_ptb_result));

    free(host_img_ori_result);
    free(host_img_ptb_result);

    cudaErrCheck(cudaDeviceReset());
    return 0;
}