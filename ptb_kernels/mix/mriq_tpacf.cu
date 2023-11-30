
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

#include "header/tpacf_header.h"
#include "kernel/tpacf_kernel.cu"

#include "mix_kernel/mriq-tpacf.cu" 

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


    // tpacf variables
    // ---------------------------------------------------------------------------------------
        // 10391
        // 97178
        // NUM_ELEMENTS = 97178;
        curandGenerator_t tpacf_gen;
        int tpacf_blks = 3;
        int tpacf_iter = 1;
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
        hist_t *tpacf_gptb_hists;
        float *tpacf_gptb_x;
        float *tpacf_gptb_y;
        float *tpacf_gptb_z;
        hist_t *host_tpacf_ori_hists;
        hist_t *host_tpacf_ptb_hists;
        hist_t *host_tpacf_gptb_hists;
    
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

        cudaErrCheck(cudaMalloc((void**) &tpacf_gptb_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMemset(tpacf_gptb_hists, 100, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t)));
        cudaErrCheck(cudaMalloc((void**) &tpacf_gptb_x, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_gptb_y, f_mem_size));
        cudaErrCheck(cudaMalloc((void**) &tpacf_gptb_z, f_mem_size));

        host_tpacf_ori_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t));
        host_tpacf_ptb_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t));
        host_tpacf_gptb_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t));

        curandErrCheck(curandCreateGenerator(&tpacf_gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(tpacf_gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(tpacf_gen, tpacf_ori_x, (1 + NUM_SETS) * num_elements));
        curandErrCheck(curandGenerateUniform(tpacf_gen, tpacf_ori_y, (1 + NUM_SETS) * num_elements));
        curandErrCheck(curandGenerateUniform(tpacf_gen, tpacf_ori_z, (1 + NUM_SETS) * num_elements));

        cudaErrCheck(cudaMemcpy(tpacf_ptb_x, tpacf_ori_x, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_ptb_y, tpacf_ori_y, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_ptb_z, tpacf_ori_z, f_mem_size, cudaMemcpyDeviceToDevice));

        cudaErrCheck(cudaMemcpy(tpacf_gptb_x, tpacf_ori_x, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_gptb_y, tpacf_ori_y, f_mem_size, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(tpacf_gptb_z, tpacf_ori_z, f_mem_size, cudaMemcpyDeviceToDevice));
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 tpacf_grid, ori_tpacf_grid;
        dim3 tpacf_block, ori_tpacf_block;
        tpacf_block.x = BLOCK_SIZE;
        tpacf_block.y = 1;
        tpacf_grid.x = NUM_SETS * 2 + 1;
        tpacf_grid.y = 1;
        ori_tpacf_grid = tpacf_grid;
        ori_tpacf_block = tpacf_block;
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

  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(272, 1, 1);
        dim3 mix_kernel_block = dim3(512, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_mriq_tpacf_kernel <<<mix_kernel_grid, mix_kernel_block>>>(numK, QGridBase, 
            mriq_gptb_x, mriq_gptb_y, mriq_gptb_z, mriq_gptb_Qr, mriq_gptb_Qi, 
            ori_mriq_grid2.x, ori_mriq_grid2.y, ori_mriq_grid2.z, ori_mriq_block2.x, ori_mriq_block2.y, ori_mriq_block2.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_mriq_grid2.x * ori_mriq_grid2.y * ori_mriq_grid2.z, tpacf_gptb_hists, tpacf_gptb_x, tpacf_gptb_y, tpacf_gptb_z, NUM_SETS, NUM_ELEMENTS, 
            ori_tpacf_grid.x, ori_tpacf_grid.y, ori_tpacf_grid.z, ori_tpacf_block.x, ori_tpacf_block.y, ori_tpacf_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_tpacf_grid.x * ori_tpacf_grid.y * ori_tpacf_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] mriq_tpacf took %f ms\n\n", kernel_time);
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
        cudaErrCheck(cudaMemcpy(host_tpacf_ori_hists, tpacf_ori_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_tpacf_gptb_hists, tpacf_gptb_hists, NUM_BINS * (NUM_SETS*2+1) * sizeof(hist_t), cudaMemcpyDeviceToHost));

        errors = 0;
        for (int i = 0; i < NUM_BINS * (NUM_SETS*2+1); i++) {
            unsigned int v1 = host_tpacf_ori_hists[i];
            unsigned int v2 = host_tpacf_gptb_hists[i];
            if (v1 - v2 != 0) {
            errors++;
            if (errors < 5) printf("%u %u\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with GENERAL VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and GENERAL VERSION agree.\n");
        }
    // ---------------------------------------------------------------------------------------

}
