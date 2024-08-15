
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

#include "header/mrif_header.h"
#include "kernel/mrif_kernel.cu"

#include "mix_kernel/cutcp-mrif.cu" 

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
        int cutcp_blks = 4;
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


    // mrif variables
    // ---------------------------------------------------------------------------------------
        int mrif_blks = 3;
        int mrif_iter = 1;
        int mrif_numX, mrif_numK;		                /* Number of X and K values */
        int original_numK;		            /* Number of K values in input file */
        float *mrif_base_kx, *mrif_base_ky, *mrif_base_kz;		        /* K trajectory (3D vectors) */
        float *mrif_base_x, *mrif_base_y, *mrif_base_z;		            /* X coordinates (3D vectors) */
        float *mrif_base_phiR, *mrif_base_phiI;		            /* Phi values (complex) */
        float *mrif_base_dR, *mrif_base_dI;		                /* D values (complex) */
        float *mrif_base_realRhoPhi, *mrif_base_imagRhoPhi;     /* RhoPhi values (complex) */
        mrif_kValues* mrif_kVals;		                /* Copy of X and RhoPhi.  Its
                                            * data layout has better cache
                                            * performance. */
        inputData(
            &original_numK, &mrif_numX,
            &mrif_base_kx, &mrif_base_ky, &mrif_base_kz,
            &mrif_base_x, &mrif_base_y, &mrif_base_z,
            &mrif_base_phiR, &mrif_base_phiI,
            &mrif_base_dR, &mrif_base_dI);
        mrif_numK = original_numK;

        // createDataStructs(mrif_numK, mrif_numX, mrif_base_realRhoPhi, mrif_base_imagRhoPhi, base_outR, base_outI);
        mrif_kVals = (mrif_kValues *)calloc(mrif_numK, sizeof (mrif_kValues));
        mrif_base_realRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));
        mrif_base_imagRhoPhi = (float* ) calloc(mrif_numK, sizeof(float));

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

        // gptb kernel
        float *gptb_x, *gptb_y, *gptb_z;
        float *gptb_outI, *gptb_outR;
        float *host_gptb_outI;		            /* Output signal (complex) */

        cudaErrCheck(cudaMalloc((void **)&ori_phiR, mrif_numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_phiI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dR, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dI, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_realRhoPhi, mrif_numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_imagRhoPhi, mrif_numK * sizeof(float)));
        // host_ori_phiMag = (float* ) memalign(16, mrif_numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(ori_phiR, mrif_base_phiR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_phiI, mrif_base_phiI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dR, mrif_base_dR, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dI, mrif_base_dI, mrif_numK * sizeof(float), cudaMemcpyHostToDevice));

        cudaErrCheck(cudaMalloc((void **)&ori_x, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_y, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ori_z, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ori_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ori_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outI, 0, mrif_numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&ptb_x, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_y, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&ptb_z, mrif_numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(ptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ptb_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ptb_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ptb_outI, 0, mrif_numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&gptb_x, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_y, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_z, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(gptb_x, mrif_base_x, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(gptb_y, mrif_base_y, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(gptb_z, mrif_base_z, mrif_numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&gptb_outR, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&gptb_outI, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(gptb_outR, 0, mrif_numX * sizeof(float)));
        cudaErrCheck(cudaMemset(gptb_outI, 0, mrif_numX * sizeof(float)));

        host_ori_outI = (float*) calloc (mrif_numX, sizeof (float));
        host_ptb_outI = (float*) calloc (mrif_numX, sizeof (float));
        host_gptb_outI = (float*) calloc (mrif_numX, sizeof (float));
    // ---------------------------------------------------------------------------------------

    // mrif kernel 1
    // ---------------------------------------------------------------------------------------
        // computeRhoPhi_GPU(mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi);
        dim3 mrif_grid1;
        dim3 mrif_block1;
        mrif_grid1.x = mrif_numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_grid1.y = 1;
        mrif_block1.x = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrif_block1.y = 1;
        printf("[ORI] mrif_grid1 -- %d * %d * %d mrif_block1 -- %d * %d * %d \n", 
                    mrif_grid1.x, mrif_grid1.y, mrif_grid1.z, mrif_block1.x, mrif_block1.y, mrif_block1.z);
        checkKernelErrors((ComputeRhoPhiGPU <<< mrif_grid1, mrif_block1 >>> (mrif_numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi)));
        cudaErrCheck(cudaMemcpy(mrif_base_realRhoPhi, ori_realRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(mrif_base_imagRhoPhi, ori_imagRhoPhi, mrif_numK * sizeof(float), cudaMemcpyDeviceToHost));

        for (int k = 0; k < mrif_numK; k++) {
            mrif_kVals[k].Kx = mrif_base_kx[k];
            mrif_kVals[k].Ky = mrif_base_ky[k];
            mrif_kVals[k].Kz = mrif_base_kz[k];
            mrif_kVals[k].RhoPhiR = mrif_base_realRhoPhi[k];
            mrif_kVals[k].RhoPhiI = mrif_base_imagRhoPhi[k];
        }
    // ---------------------------------------------------------------------------------------
    
    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 mrif_grid2, ori_mrif_grid2;
        dim3 mrif_block2, ori_mrif_block2;
        mrif_grid2.x = mrif_numX / KERNEL_FH_THREADS_PER_BLOCK;
        mrif_grid2.y = 1;
        mrif_block2.x = KERNEL_FH_THREADS_PER_BLOCK;
        mrif_block2.y = 1;
        ori_mrif_grid2 = mrif_grid2;
        ori_mrif_block2 = mrif_block2;
        printf("[ORI] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \n", 
                    mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

        int FHGridBase = 0 * KERNEL_FH_K_ELEMS_PER_GRID;
        mrif_kValues* mrif_kValsTile = mrif_kVals + FHGridBase;
        int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, mrif_numK - FHGridBase);
        cudaMemcpyToSymbol(c, mrif_kValsTile, numElems * sizeof(mrif_kValues), 0);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_mrif <<< mrif_grid2, mrif_block2 >>> (
                mrif_numK, FHGridBase, ori_x, ori_y, ori_z, ori_outR, ori_outI, mrif_iter)));
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
        checkKernelErrors((ptb_mrif <<< mrif_grid2, mrif_block2 >>> (mrif_numK, FHGridBase, ptb_x, ptb_y, ptb_z, ptb_outR, ptb_outI, 
                                    mrif_grid2_dim_x, mrif_block2_dim_x,
                                    mrif_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] mrif took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(272, 1, 1);
        dim3 mix_kernel_block = dim3(384, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_cutcp_mrif_kernel <<<mix_kernel_grid, mix_kernel_block>>>(binDim_x, binDim_y, cutcp_gptb_binZeroCuda,
    h, cutoff2, inv_cutoff2, cutcp_gptb_regionZeroCuda, 25, 
    ori_cutcp_grid.x, ori_cutcp_grid.y, ori_cutcp_grid.z, ori_cutcp_block.x, ori_cutcp_block.y, ori_cutcp_block.z,
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cutcp_grid.x * ori_cutcp_grid.y * ori_cutcp_grid.z, mrif_numK, FHGridBase, gptb_x, gptb_y, gptb_z, gptb_outR, gptb_outI, 
            ori_mrif_grid2.x, ori_mrif_grid2.y, ori_mrif_grid2.z, ori_mrif_block2.x, ori_mrif_block2.y, ori_mrif_block2.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_mrif_grid2.x * ori_mrif_grid2.y * ori_mrif_grid2.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cutcp_mrif took %f ms\n\n", kernel_time);
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
        cudaErrCheck(cudaMemcpy(host_ori_outI, ori_outI, mrif_numX * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_gptb_outI, gptb_outI, mrif_numX * sizeof(float), cudaMemcpyDeviceToHost));
        errors = 0;
        for (int i = 0; i < mrif_numX; i++) {
            float v1 = host_ori_outI[i];
            float v2 = host_gptb_outI[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 5) printf("%f %f\n", v1, v2);
            }
            if (i < 3) printf("%d %f %f\n", i, v1, v2);
        }
        if (errors > 0) {
            printf("ORIGIN VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIGIN VERSION and GPTB VERSION agree.\n");
        }
    // ---------------------------------------------------------------------------------------

}
