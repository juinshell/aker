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

#include "header/tpacf_header.h"
#include "file_t/tpacf_kernel.cu"

int main(int argc, char* argv[]) {
    int tpacf_blks = 3;
    int tpacf_iter = 1;
    if (argc == 3) {
        tpacf_blks = atoi(argv[1]);
        tpacf_iter = atoi(argv[2]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
        float kernel_time;
        curandGenerator_t gen;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    // tpacf variables
    // ---------------------------------------------------------------------------------------
        // 10391
        // 97178
        // NUM_ELEMENTS = 97178;
        int NUM_ELEMENTS = 4096;
        int NUM_SETS = 100;
        int num_elements = NUM_ELEMENTS; 
        unsigned f_mem_size = (1 + NUM_SETS) * num_elements * sizeof(float);
        float *binb = (float *)malloc((NUM_BINS+1)*sizeof(float));
        for (int k = 0; k < NUM_BINS+1; k++){
            binb[k] = cos(pow(10.0, (log10(min_arcmin) + k*1.0/bins_per_dec)) / 60.0*D2R);
        }

        hist_t *tpacf_ori_hists;
        float *tpacf_ori_x;
        float *tpacf_ori_y;
        float *tpacf_ori_z;
        hist_t *tpacf_ptb_hists;
        float *tpacf_ptb_x;
        float *tpacf_ptb_y;
        float *tpacf_ptb_z;
        hist_t *host_tpacf_ori_hists;
        hist_t *host_tpacf_ptb_hists;
    
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMemset(tpacf_ori_hists, 100, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_x, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_y, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ori_z, f_mem_size));

        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMemset(tpacf_ptb_hists, 100, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_x, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_y, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_ptb_z, f_mem_size));

        host_tpacf_ori_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t));
        host_tpacf_ptb_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t));

        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(gen, tpacf_ori_x, (1 + NUM_SETS) * num_elements));
        curandErrCheck(curandGenerateUniform(gen, tpacf_ori_y, (1 + NUM_SETS) * num_elements));
        curandErrCheck(curandGenerateUniform(gen, tpacf_ori_z, (1 + NUM_SETS) * num_elements));

        cudaErrCheck(cudaMemcpy(tpacf_ptb_x, tpacf_ori_x, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_ptb_y, tpacf_ori_y, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_ptb_z, tpacf_ori_z, f_mem_size, cudaMemcpyDeviceToDevice));
    // ---------------------------------------------------------------------------------------
    

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 tpacf_grid;
        dim3 tpacf_block;
        tpacf_block.x = BLOCK_SIZE;
        tpacf_block.y = 1;
        tpacf_grid.x = NUM_SETS * 2 + 1;
        tpacf_grid.y = 1;
        printf("[ORI] Running with tpacf...\n");
        printf("[ORI] tpacf_grid -- %d * %d * %d tpacf_block -- %d * %d * %d \n", 
                tpacf_grid.x, tpacf_grid.y, tpacf_grid.z, tpacf_block.x, tpacf_block.y, tpacf_block.z);
        
        cudaMemcpyToSymbol(dev_binb, binb, (NUM_BINS+1)*sizeof(float));
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_tpacf <<< tpacf_grid, tpacf_block >>> (tpacf_ori_hists, tpacf_ori_x, tpacf_ori_y, tpacf_ori_z, 
                            NUM_SETS, NUM_ELEMENTS, tpacf_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] tpacf took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    // PTB
    // ---------------------------------------------------------------------------------
        int tpacf_grid_dim_x = tpacf_grid.x;
        int tpacf_grid_dim_y = tpacf_grid.y;
        int tpacf_block_dim_x = tpacf_block.x;
        int tpacf_block_dim_y = tpacf_block.y;
        tpacf_grid.x = tpacf_blks == 0 ? tpacf_grid_dim_x * tpacf_grid_dim_y : SM_NUM * tpacf_blks;
        tpacf_grid.y = 1;
        tpacf_block.x = tpacf_block_dim_x * tpacf_block_dim_y;
        tpacf_block.y = 1;
        printf("[PTB] Running with tpacf...\n");
        printf("[PTB] tpacf_grid -- %d * %d * %d tpacf_block -- %d * %d * %d \n", 
            tpacf_grid.x, tpacf_grid.y, tpacf_grid.z, tpacf_block.x, tpacf_block.y, tpacf_block.z);
        
        cudaMemcpyToSymbol(dev_binb, binb, (NUM_BINS+1)*sizeof(float));
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_tpacf <<< tpacf_grid, tpacf_block >>> (tpacf_ptb_hists, tpacf_ptb_x, tpacf_ptb_y, tpacf_ptb_z, 
                            NUM_SETS, NUM_ELEMENTS, tpacf_grid_dim_x, tpacf_grid_dim_y, tpacf_block_dim_x, tpacf_block_dim_y, tpacf_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] tpacf took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
        printf("Checking results...\n");
        cudaErrCheck(cudaMemcpy(host_tpacf_ori_hists, tpacf_ori_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_tpacf_ptb_hists, tpacf_ptb_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int i = 0; i < NUM_BINS * (NUM_SETS*2+1); i++) {
            unsigned int v1 = host_tpacf_ori_hists[i];
            unsigned int v2 = host_tpacf_ptb_hists[i];
            if (v1 - v2 != 0) {
            errors++;
            if (errors < 5) printf("%u %u\n", v1, v2);
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

    cudaErrCheck(cudaDeviceReset());
    return 0;
}