
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

#include "header/tpacf_header.h"
#include "kernel/tpacf_kernel.cu"

#include "mix_kernel/cutcp-tpacf.cu" 

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
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(1024, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_cutcp_tpacf_kernel_6_1 <<<mix_kernel_grid, mix_kernel_block>>>(binDim_x, binDim_y, cutcp_gptb_binZeroCuda,
    h, cutoff2, inv_cutoff2, cutcp_gptb_regionZeroCuda, 25, 
    ori_cutcp_grid.x, ori_cutcp_grid.y, ori_cutcp_grid.z, ori_cutcp_block.x, ori_cutcp_block.y, ori_cutcp_block.z,
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cutcp_grid.x * ori_cutcp_grid.y * ori_cutcp_grid.z, tpacf_gptb_hists, tpacf_gptb_x, tpacf_gptb_y, tpacf_gptb_z, NUM_SETS, NUM_ELEMENTS, 
            ori_tpacf_grid.x, ori_tpacf_grid.y, ori_tpacf_grid.z, ori_tpacf_block.x, ori_tpacf_block.y, ori_tpacf_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_tpacf_grid.x * ori_tpacf_grid.y * ori_tpacf_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cutcp_tpacf_6_1 took %f ms\n\n", kernel_time);
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
