
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

#include <mma.h>
using namespace nvcuda; 
#include "header/atom.h"
#include "header/cutcp_header.h"
#include "kernel/cutcp_kernel.cu"

#include "header/mriq_header.h"
#include "kernel/mriq_kernel.cu"

#include "mix_kernel/cutcp-mriq.cu" 

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

    // cutcp variables
    // ---------------------------------------------------------------------------------------
        int cutcp_blks = 6;
        int cutcp_iter = 1;
        Atoms *atom;
        LatticeDim lattice_dim;
        Lattice *gpu_lattice;
        Vec3 min_ext, max_ext;	    /* Bounding box of atoms */
        Vec3 lo, hi;			    /* Bounding box with padding  */
        float h = 0.5f;		        /* Lattice spacing */
        float cutoff = 12.f;		/* Cutoff radius */
        float padding = 0.5f;		/* Bounding box padding distance */
        
        const char *pqrfilename = "/home/jxdeng/workspace/tacker/0_mybench/file_t/cutcp_input.pqr";
        if (!(atom = read_atom_file(pqrfilename))) {
            fprintf(stderr, "read_atom_file() failed\n");
            exit(1);
        }
        get_atom_extent(&min_ext, &max_ext, atom);
        lo = (Vec3) {min_ext.x - padding, min_ext.y - padding, min_ext.z - padding};
        hi = (Vec3) {max_ext.x + padding, max_ext.y + padding, max_ext.z + padding};
        lattice_dim = lattice_from_bounding_box(lo, hi, h);
        gpu_lattice = create_lattice(lattice_dim);

        float4 *binBaseAddr;
        int3 *nbrlist;
        nbrlist = (int3 *)malloc(NBRLIST_MAXLEN * sizeof(int3));
        int nbins = 32768;
        binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
        prepare_input(gpu_lattice, cutoff, atom, binBaseAddr, nbrlist);

        int nbrlistlen = 256;
        float *cutcp_ori_regionZeroCuda, *host_cutcp_ori_regionZeroCuda;
        float4 *cutcp_ori_binBaseCuda, *cutcp_ori_binZeroCuda;
        float *cutcp_ptb_regionZeroCuda, *host_cutcp_ptb_regionZeroCuda;
        float4 *cutcp_ptb_binBaseCuda, *cutcp_ptb_binZeroCuda;
        float *cutcp_gptb_regionZeroCuda, *host_cutcp_gptb_regionZeroCuda;
        float4 *cutcp_gptb_binBaseCuda, *cutcp_gptb_binZeroCuda;

        int lnx = 208;
        int lny = 208;
        int lnz = 208;
        int lnall = lnx * lny * lnz;

        int xRegionDim = 26;
        int yRegionDim = 26;
        int zRegionDim = 26;
        int binDim_x = 32;
        int binDim_y = 32;
        float cutoff2 = 144.0;
        float inv_cutoff2 = 0.006944;

        cudaErrCheck(cudaMalloc((void **) &cutcp_ori_regionZeroCuda, lnall * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **) &cutcp_ptb_regionZeroCuda, lnall * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **) &cutcp_gptb_regionZeroCuda, lnall * sizeof(float)));
        cudaErrCheck(cudaMemset(cutcp_ori_regionZeroCuda, 0, lnall * sizeof(float)));
        cudaErrCheck(cudaMemset(cutcp_ptb_regionZeroCuda, 0, lnall * sizeof(float)));
        cudaErrCheck(cudaMemset(cutcp_gptb_regionZeroCuda, 0, lnall * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **) &cutcp_ori_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
        cudaErrCheck(cudaMalloc((void **) &cutcp_ptb_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
        cudaErrCheck(cudaMalloc((void **) &cutcp_gptb_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
        cudaErrCheck(cudaMemcpy(cutcp_ori_binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
            cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(cutcp_ptb_binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
            cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(cutcp_gptb_binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
            cudaMemcpyHostToDevice));

        cutcp_ori_binZeroCuda = cutcp_ori_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;
        cutcp_ptb_binZeroCuda = cutcp_ptb_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;
        cutcp_gptb_binZeroCuda = cutcp_gptb_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;

        host_cutcp_ori_regionZeroCuda = (float *)malloc(lnall * sizeof(float));
        host_cutcp_ptb_regionZeroCuda = (float *)malloc(lnall * sizeof(float));
        host_cutcp_gptb_regionZeroCuda = (float *)malloc(lnall * sizeof(float));

        cudaErrCheck(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
        cudaErrCheck(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 cutcp_grid, cutcp_block, ori_cutcp_grid, ori_cutcp_block;
        cutcp_grid.x = xRegionDim;
        cutcp_grid.y = yRegionDim;
        cutcp_grid.z = cutcp_iter * 2;
        cutcp_block.x = 8;
        cutcp_block.y = 2;
        cutcp_block.z = 8;
        ori_cutcp_grid = cutcp_grid;
        ori_cutcp_block = cutcp_block;


        printf("[ORI] Running with cutcp...\n");
        printf("[ORI] cutcp_grid -- %d * %d * %d cutcp_block -- %d * %d * %d \n", 
            cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, cutcp_block.x, cutcp_block.y, cutcp_block.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_ori_binZeroCuda, h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, 1)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] cutcp took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    // PTB
    // ---------------------------------------------------------------------------------
        int cutcp_grid_dim_x = cutcp_grid.x;
        int cutcp_grid_dim_y = cutcp_grid.y;
        int cutcp_grid_dim_z = cutcp_grid.z;
        cutcp_grid.x = cutcp_blks == 0 ? cutcp_grid_dim_x * cutcp_grid_dim_y * cutcp_grid_dim_z : SM_NUM * cutcp_blks;
        cutcp_grid.y = 1;
        cutcp_grid.z = 1;

        cudaErrCheck(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
	    cudaErrCheck(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));

        printf("[PTB] Running with cutcp...\n");
        printf("[PTB] cutcp_grid -- %d * %d * %d cutcp_block -- %d * %d * %d \n", 
            cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, cutcp_block.x, cutcp_block.y, cutcp_block.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb2_cutcp<<<cutcp_grid, cutcp_block>>>(
            binDim_x, binDim_y, cutcp_ptb_binZeroCuda, 
            h, cutoff2, inv_cutoff2, cutcp_ptb_regionZeroCuda, 25, 
            cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z, 1)));

        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] cutcp took %f ms\n\n", kernel_time);
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


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(1024, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_cutcp_mriq_kernel_6_1 <<<mix_kernel_grid, mix_kernel_block>>>(binDim_x, binDim_y, cutcp_gptb_binZeroCuda,
    h, cutoff2, inv_cutoff2, cutcp_gptb_regionZeroCuda, 25, 
    ori_cutcp_grid.x, ori_cutcp_grid.y, ori_cutcp_grid.z, ori_cutcp_block.x, ori_cutcp_block.y, ori_cutcp_block.z,
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cutcp_grid.x * ori_cutcp_grid.y * ori_cutcp_grid.z, numK, QGridBase, 
            mriq_gptb_x, mriq_gptb_y, mriq_gptb_z, mriq_gptb_Qr, mriq_gptb_Qi, 
            ori_mriq_grid2.x, ori_mriq_grid2.y, ori_mriq_grid2.z, ori_mriq_block2.x, ori_mriq_block2.y, ori_mriq_block2.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_mriq_grid2.x * ori_mriq_grid2.y * ori_mriq_grid2.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cutcp_mriq 6_1 took %f ms\n\n", kernel_time);
  // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
        cudaErrCheck(cudaMemcpy(host_cutcp_ori_regionZeroCuda, cutcp_ori_regionZeroCuda, lnall * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_cutcp_gptb_regionZeroCuda, cutcp_gptb_regionZeroCuda, lnall * sizeof(float), cudaMemcpyDeviceToHost));
        
        errors = 0;
        for (int i = 0; i < lnall; i++) {
            float v1 = host_cutcp_ori_regionZeroCuda[i];
            float v2 = host_cutcp_gptb_regionZeroCuda[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 10) printf("%f %f\n", v1, v2);
            }
            if (i < 3) printf("%f %f\n", v1, v2);
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
        }


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

}
