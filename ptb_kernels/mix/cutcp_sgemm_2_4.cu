
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

#include "header/sgemm_header.h"
#include "kernel/sgemm_kernel.cu"

#include "mix_kernel/cutcp-sgemm.cu" 

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
        dim3 mix_kernel_grid = dim3(68, 1, 1);
        dim3 mix_kernel_block = dim3(768, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_cutcp_sgemm_kernel_2_4 <<<mix_kernel_grid, mix_kernel_block>>>(binDim_x, binDim_y, cutcp_gptb_binZeroCuda,
    h, cutoff2, inv_cutoff2, cutcp_gptb_regionZeroCuda, 25, 
    ori_cutcp_grid.x, ori_cutcp_grid.y, ori_cutcp_grid.z, ori_cutcp_block.x, ori_cutcp_block.y, ori_cutcp_block.z,
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cutcp_grid.x * ori_cutcp_grid.y * ori_cutcp_grid.z, sgemm_gptb_a, sgemm_gptb_b, sgemm_gptb_c, NORMAL_M, NORMAL_N, NORMAL_K,
            ori_sgemm_grid.x, ori_sgemm_grid.y, ori_sgemm_grid.z, ori_sgemm_block.x, ori_sgemm_block.y, ori_sgemm_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_sgemm_grid.x * ori_sgemm_grid.y * ori_sgemm_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cutcp_sgemm 2_4 took %f ms\n\n", kernel_time);
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
