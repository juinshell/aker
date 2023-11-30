#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
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


#include "header/mrif_header.h"
#include "file_t/mrif_kernel.cu"

// original PTB -- 2
// elastic PTB -- 3

int main (int argc, char *argv[]) {
    int mrif_blks = 3;
    int mrif_iter = 1;
    if (argc == 3) {
        mrif_blks = atoi(argv[1]);
        mrif_iter = atoi(argv[2]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
        float kernel_time;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // ---------------------------------------------------------------------------------------

    // mrif variables
    // ---------------------------------------------------------------------------------------
        int numX, numK;		                /* Number of X and K values */
        int original_numK;		            /* Number of K values in input file */
        float *base_kx, *base_ky, *base_kz;		        /* K trajectory (3D vectors) */
        float *base_x, *base_y, *base_z;		            /* X coordinates (3D vectors) */
        float *base_phiR, *base_phiI;		            /* Phi values (complex) */
        float *base_dR, *base_dI;		                /* D values (complex) */
        float *base_realRhoPhi, *base_imagRhoPhi;     /* RhoPhi values (complex) */
        kValues* kVals;		                /* Copy of X and RhoPhi.  Its
                                            * data layout has better cache
                                            * performance. */
        inputData(
            &original_numK, &numX,
            &base_kx, &base_ky, &base_kz,
            &base_x, &base_y, &base_z,
            &base_phiR, &base_phiI,
            &base_dR, &base_dI);
        numK = original_numK;

        // createDataStructs(numK, numX, base_realRhoPhi, base_imagRhoPhi, base_outR, base_outI);
        kVals = (kValues *)calloc(numK, sizeof (kValues));
        base_realRhoPhi = (float* ) calloc(numK, sizeof(float));
        base_imagRhoPhi = (float* ) calloc(numK, sizeof(float));

        // kernel 1
        float *ori_phiR, *ori_phiI;
        float *ori_dR, *ori_dI;
        float *ori_realRhoPhi, *ori_imagRhoPhi;
        // kernel 2
        float *ori_x, *ori_y, *ori_z;
        float *ori_outI, *ori_outR;
        float *host_ori_outI;		            /* Output signal (complex) */

        // kernel 2
        float *ptb_x, *ptb_y, *ptb_z;
        float *ptb_outI, *ptb_outR;
        float *host_ptb_outI;		            /* Output signal (complex) */

        cudaErrCheck(cudaMalloc((void **)&ori_phiR, numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_phiI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dR, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_realRhoPhi, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_imagRhoPhi, numK * sizeof(float)));
        // host_ori_phiMag = (float* ) memalign(16, numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(ori_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dR, base_dR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dI, base_dI, numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&ori_x, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_y, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_z, numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ori_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ori_outR, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_outI, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outR, 0, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outI, 0, numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&ptb_x, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_y, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_z, numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ptb_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ptb_outR, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ptb_outI, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outR, 0, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outI, 0, numX * sizeof(float)));

        host_ori_outI = (float*) calloc (numX, sizeof (float));
        host_ptb_outI = (float*) calloc (numX, sizeof (float));
    // ---------------------------------------------------------------------------------------

    // mrif kernel 1
    // ---------------------------------------------------------------------------------------
        // computeRhoPhi_GPU(numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi);
        dim3 mrif_grid1;
        dim3 mrif_block1;
        mrif_grid1.x = numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_grid1.y = 1;
        mrif_block1.x = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_block1.y = 1;
        printf("[ORI] mrif_grid1 -- %d * %d * %d mrif_block1 -- %d * %d * %d \n", 
                    mrif_grid1.x, mrif_grid1.y, mrif_grid1.z, mrif_block1.x, mrif_block1.y, mrif_block1.z);
        checkKernelErrors((ComputeRhoPhiGPU <<< mrif_grid1, mrif_block1 >>> (numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi)));
        cudaErrCheck(cudaMemcpy(base_realRhoPhi, ori_realRhoPhi, numK * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(base_imagRhoPhi, ori_imagRhoPhi, numK * sizeof(float), cudaMemcpyDeviceToHost));

        for (int k = 0; k < numK; k++) {
            kVals[k].Kx = base_kx[k];
            kVals[k].Ky = base_ky[k];
            kVals[k].Kz = base_kz[k];
            kVals[k].RhoPhiR = base_realRhoPhi[k];
            kVals[k].RhoPhiI = base_imagRhoPhi[k];
        }
    // ---------------------------------------------------------------------------------------


    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 mrif_grid2;
        dim3 mrif_block2;
        mrif_grid2.x = numX / KERNEL_FH_THREADS_PER_BLOCK;
        mrif_grid2.y = 1;
        mrif_block2.x = KERNEL_FH_THREADS_PER_BLOCK;
        mrif_block2.y = 1;
        printf("[ORI] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \n", 
                    mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        int FHGridBase = 0 * KERNEL_FH_K_ELEMS_PER_GRID;
        kValues* kValsTile = kVals + FHGridBase;
        int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, numK - FHGridBase);
        cudaMemcpyToSymbol(c, kValsTile, numElems * sizeof(kValues), 0);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_mrif <<< mrif_grid2, mrif_block2 >>> (
                numK, FHGridBase, ori_x, ori_y, ori_z, ori_outR, ori_outI, mrif_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] mrif took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    // PTB running
    // ---------------------------------------------------------------------------------------
        int mrif_grid2_dim_x = mrif_grid2.x;
        int mrif_block2_dim_x = mrif_block2.x;
        mrif_grid2.x = mrif_blks == 0 ? mrif_grid2_dim_x : SM_NUM * mrif_blks;
        printf("[PTB] Running with mrif...\n");
        printf("[PTB] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \n", 
            mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_mrif <<< mrif_grid2, mrif_block2 >>> (numK, FHGridBase, ptb_x, ptb_y, ptb_z, ptb_outR, ptb_outI, 
                                    mrif_grid2_dim_x, mrif_block2_dim_x,
                                    mrif_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] mrif took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
        cudaErrCheck(cudaMemcpy(host_ori_outI, ori_outI, numX * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_ptb_outI, ptb_outI, numX * sizeof(float), cudaMemcpyDeviceToHost));
        int errors = 0;
        for (int i = 0; i < numX; i++) {
            float v1 = host_ori_outI[i];
            float v2 = host_ptb_outI[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 5) printf("%f %f\n", v1, v2);
            }
            if (i < 3) printf("%d %f %f\n", i, v1, v2);
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

