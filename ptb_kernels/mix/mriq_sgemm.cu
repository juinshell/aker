
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

#include "header/mriq_header.h"
#include "kernel/mriq_kernel.cu"

#include "header/sgemm_header.h"
#include "kernel/sgemm_kernel.cu"

#include "mix_kernel/mriq-sgemm.cu" 

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

    // mriq variables
    // ---------------------------------------------------------------------------------------
        int mriq_blks = 4;
        int mriq_iter = 1;
        int numK = 2097152;
        int numX = 2097152;
        float *base_kx, *base_ky, *base_kz;		/* K trajectory (3D vectors) */
        float *base_x, *base_y, *base_z;		/* X coordinates (3D vectors) */
        float *base_phiR, *base_phiI;		    /* Phi values (complex) */
        // float *base_phiMag;		                /* Magnitude of Phi */
        // float *base_Qr, *base_Qi;		        /* Q signal (complex) */
        struct mriq_kValues* mriq_kVals;

        // kernel 1
        float *mriq_ori_phiR, *mriq_ori_phiI;
        float *mriq_ori_phiMag, *host_mriq_ori_phiMag;
        // kernel 2
        float *mriq_ori_x, *mriq_ori_y, *mriq_ori_z;
        float *mriq_ori_Qr, *mriq_ori_Qi, *host_mriq_ori_Qi;

        // // kernel 1
        // float *ptb_phiR, *ptb_phiI;
        // float *ptb_phiMag, *host_ptb_phiMag;
        // kernel 2
        float *mriq_ptb_x, *mriq_ptb_y, *mriq_ptb_z;
        float *mriq_ptb_Qr, *mriq_ptb_Qi, *host_mriq_ptb_Qi;

        // gptb kernel 2
        float *mriq_gptb_x, *mriq_gptb_y, *mriq_gptb_z;
        float *mriq_gptb_Qr, *mriq_gptb_Qi, *host_mriq_gptb_Qi;

        inputData(&numK, &numX,
            &base_kx, &base_ky, &base_kz,
            &base_x, &base_y, &base_z,
            &base_phiR, &base_phiI);
        numK = 2097152;

        // Memory allocation
        // base_phiMag = (float* ) memalign(16, numK * sizeof(float));
        // base_Qr = (float*) memalign(16, numX * sizeof (float));
        // base_Qi = (float*) memalign(16, numX * sizeof (float));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiR, numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiMag, numK * sizeof(float)));
        host_mriq_ori_phiMag = (float* ) memalign(16, numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(mriq_ori_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ori_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&mriq_ori_x, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_y, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_z, numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(mriq_ori_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ori_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ori_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_Qr, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ori_Qi, numX * sizeof(float)));
        cudaMemset((void *)mriq_ori_Qr, 0, numX * sizeof(float));
        cudaMemset((void *)mriq_ori_Qi, 0, numX * sizeof(float));
        host_mriq_ori_Qi = (float*) memalign(16, numX * sizeof (float));

        // cudaErrCheck(cudaMalloc((void **)&ptb_phiR, numK * sizeof(float)));   
        // cudaErrCheck(cudaMalloc((void **)&ptb_phiI, numK * sizeof(float)));
        // cudaErrCheck(cudaMalloc((void **)&ptb_phiMag, numK * sizeof(float)));
        // host_ptb_phiMag = (float* ) memalign(16, numK * sizeof(float));
        // cudaErrCheck(cudaMemcpy(ptb_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
        // cudaErrCheck(cudaMemcpy(ptb_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&mriq_ptb_x, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ptb_y, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ptb_z, numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(mriq_ptb_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ptb_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_ptb_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&mriq_ptb_Qr, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_ptb_Qi, numX * sizeof(float)));
        cudaMemset((void *)mriq_ptb_Qr, 0, numX * sizeof(float));
        cudaMemset((void *)mriq_ptb_Qi, 0, numX * sizeof(float));
        host_mriq_ptb_Qi = (float*) memalign(16, numX * sizeof (float));

        // gptb
        cudaErrCheck(cudaMalloc((void **)&mriq_gptb_x, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_gptb_y, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_gptb_z, numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(mriq_gptb_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_gptb_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(mriq_gptb_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&mriq_gptb_Qr, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&mriq_gptb_Qi, numX * sizeof(float)));
        cudaMemset((void *)mriq_gptb_Qr, 0, numX * sizeof(float));
        cudaMemset((void *)mriq_gptb_Qi, 0, numX * sizeof(float));
        host_mriq_gptb_Qi = (float*) memalign(16, numX * sizeof (float));
    // ---------------------------------------------------------------------------------------

    // PRE running
    // ---------------------------------------------------------------------------------------
        dim3 mriq_grid1;
        dim3 mriq_block1;
        mriq_grid1.x = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
        mriq_grid1.y = 1;
        mriq_block1.x = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
        mriq_block1.y = 1;
        printf("[ORI] Running with mriq...\n");
        printf("[ORI] mriq_grid1 -- %d * %d * %d mriq_block1 -- %d * %d * %d \n", 
            mriq_grid1.x, mriq_grid1.y, mriq_grid1.z, mriq_block1.x, mriq_block1.y, mriq_block1.z);

        checkKernelErrors((ori_ComputePhiMag <<< mriq_grid1, mriq_block1 >>> (mriq_ori_phiR, mriq_ori_phiI, mriq_ori_phiMag, numK)));
        cudaMemcpy(host_mriq_ori_phiMag, mriq_ori_phiMag, numK * sizeof(float), cudaMemcpyDeviceToHost);

        mriq_kVals = (struct mriq_kValues*)calloc(numK, sizeof (struct mriq_kValues));
        for (int k = 0; k < numK; k++) {
            mriq_kVals[k].Kx = base_kx[k];
            mriq_kVals[k].Ky = base_ky[k];
            mriq_kVals[k].Kz = base_kz[k];
            mriq_kVals[k].PhiMag = host_mriq_ori_phiMag[k];
        }
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        numX = (numX / 10) * mriq_iter;

        dim3 mriq_grid2, ori_mriq_grid2;
        dim3 mriq_block2, ori_mriq_block2;
        mriq_grid2.x = numX / KERNEL_Q_THREADS_PER_BLOCK;
        mriq_grid2.y = 1;
        mriq_block2.x = KERNEL_Q_THREADS_PER_BLOCK;
        mriq_block2.y = 1;
        ori_mriq_grid2 = mriq_grid2;
        ori_mriq_block2 = mriq_block2;
        printf("[ORI] mriq_grid2 -- %d * %d * %d mriq_block2 -- %d * %d * %d \n", 
            mriq_grid2.x, mriq_grid2.y, mriq_grid2.z, mriq_block2.x, mriq_block2.y, mriq_block2.z);

        int QGridBase = 0 * KERNEL_Q_K_ELEMS_PER_GRID;
        mriq_kValues* kValsTile = mriq_kVals + QGridBase;
        cudaMemcpyToSymbol(ck, kValsTile, KERNEL_Q_K_ELEMS_PER_GRID * sizeof(mriq_kValues), 0);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_mriq <<< mriq_grid2, mriq_block2 >>>(numK, QGridBase, 
                                mriq_ori_x, mriq_ori_y, mriq_ori_z, mriq_ori_Qr, mriq_ori_Qi, 
                                1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] mriq took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        int mriq_grid2_dim_x = mriq_grid2.x;
        // int mriq_block2_dim_x = mriq_block2.x;
        mriq_grid2.x = SM_NUM * 2;
        mriq_grid2.x = mriq_blks == 0 ? mriq_grid2_dim_x : SM_NUM * mriq_blks;
        printf("[PTB] Running with mriq...\n");
        printf("[PTB] mriq_grid2 -- %d * %d * %d mriq_block2 -- %d * %d * %d \n", 
            mriq_grid2.x, mriq_grid2.y, mriq_grid2.z, mriq_block2.x, mriq_block2.y, mriq_block2.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb2_mriq <<< mriq_grid2, mriq_block2 >>>(numK, QGridBase, 
                                mriq_ptb_x, mriq_ptb_y, mriq_ptb_z, mriq_ptb_Qr, mriq_ptb_Qi, 
                                mriq_grid2_dim_x, 
                                1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] mriq took %f ms\n\n", kernel_time);
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
        dim3 mix_kernel_block = dim3(384, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_mriq_sgemm_kernel <<<mix_kernel_grid, mix_kernel_block>>>(numK, QGridBase, 
            mriq_gptb_x, mriq_gptb_y, mriq_gptb_z, mriq_gptb_Qr, mriq_gptb_Qi, 
            ori_mriq_grid2.x, ori_mriq_grid2.y, ori_mriq_grid2.z, ori_mriq_block2.x, ori_mriq_block2.y, ori_mriq_block2.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_mriq_grid2.x * ori_mriq_grid2.y * ori_mriq_grid2.z, sgemm_gptb_a, sgemm_gptb_b, sgemm_gptb_c, NORMAL_M, NORMAL_N, NORMAL_K,
            ori_sgemm_grid.x, ori_sgemm_grid.y, ori_sgemm_grid.z, ori_sgemm_block.x, ori_sgemm_block.y, ori_sgemm_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] mriq_sgemm took %f ms\n\n", kernel_time);
  // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
        cudaMemcpy(host_mriq_ori_Qi, mriq_ori_Qi, numK * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_mriq_gptb_Qi, mriq_gptb_Qi, numK * sizeof(float), cudaMemcpyDeviceToHost);
                errors = 0;
        for (int i = 0; i < numX; i++) {
            float v1 = host_mriq_ori_Qi[i];
            float v2 = host_mriq_gptb_Qi[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 5) printf("%f %f\n", v1, v2);
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
