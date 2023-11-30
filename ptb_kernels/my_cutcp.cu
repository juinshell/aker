#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>
#include "header/atom.h"

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

#include "header/cutcp_header.h"
#include "kernel/cutcp_kernel.cu"


__global__ void mix_cutcp(
    int binDim_x,
    int binDim_y,
    float4 *binZeroAddr,    /* address of atom bins starting at origin */
    float h,                /* lattice spacing */
    float cutoff2,          /* square of cutoff distance */
    float inv_cutoff2,
    float *regionZeroAddr, /* address of lattice regions starting at origin */
    int zRegionIndex_t,
    int cutcp_grid_dim_x, int cutcp_grid_dim_y, int cutcp_grid_dim_z,
    int cutcp_block_dim_x, int cutcp_block_dim_y, int cutcp_block_dim_z,
    int cutcp_iter){
    int thread_step = 0;
    if (threadIdx.x < 128) {
        thread_step = 0;
        mix_cutcp0(
            binDim_x,
            binDim_y,
            binZeroAddr,    /* address of atom bins starting at origin */
            h,                /* lattice spacing */
            cutoff2,          /* square of cutoff distance */
            inv_cutoff2,
            regionZeroAddr, /* address of lattice regions starting at origin */
            zRegionIndex_t,
            cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
            cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z,
            thread_step, cutcp_iter
            );
    } else if (threadIdx.x < 128 * 2) {
        thread_step = 128 * 1;
        mix_cutcp1(
            binDim_x,
            binDim_y,
            binZeroAddr,    /* address of atom bins starting at origin */
            h,                /* lattice spacing */
            cutoff2,          /* square of cutoff distance */
            inv_cutoff2,
            regionZeroAddr, /* address of lattice regions starting at origin */
            zRegionIndex_t,
            cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
            cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z,
            thread_step, cutcp_iter
            );
    } else if (threadIdx.x < 128 * 3){
        thread_step = 128 * 2;
        mix_cutcp2(
            binDim_x,
            binDim_y,
            binZeroAddr,    /* address of atom bins starting at origin */
            h,                /* lattice spacing */
            cutoff2,          /* square of cutoff distance */
            inv_cutoff2,
            regionZeroAddr, /* address of lattice regions starting at origin */
            zRegionIndex_t,
            cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
            cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z,
            thread_step, cutcp_iter
            );
    }
}


int main(int argc, char* argv[]) {
    int cutcp_blks = 6;
    int cutcp_iter = 1;
    if (argc == 3) {
        cutcp_blks = atoi(argv[1]);
        cutcp_iter = atoi(argv[2]);
    }

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
        Atoms *atom;
        LatticeDim lattice_dim;
        Lattice *gpu_lattice;
        Vec3 min_ext, max_ext;	    /* Bounding box of atoms */
        Vec3 lo, hi;			    /* Bounding box with padding  */
        float h = 0.5f;		        /* Lattice spacing */
        float cutoff = 12.f;		/* Cutoff radius */
        // float exclcutoff = 1.f;	    /* Radius for exclusion */
        float padding = 0.5f;		/* Bounding box padding distance */
        
        const char *pqrfilename = "/home/jxdeng/workspace/tacker/0_mybench/file_t/cutcp_input.pqr";
        if (!(atom = read_atom_file(pqrfilename))) {
            fprintf(stderr, "read_atom_file() failed\n");
            exit(1);
        }
        // printf("read %d atoms from file '%s'\n", atom->size, pqrfilename);
        get_atom_extent(&min_ext, &max_ext, atom);
        // printf("extent of domain is:\n");
        // printf("  minimum %g %g %g\n", min_ext.x, min_ext.y, min_ext.z);
        // printf("  maximum %g %g %g\n", max_ext.x, max_ext.y, max_ext.z);
        // printf("padding domain by %g Angstroms\n", padding);
        lo = (Vec3) {min_ext.x - padding, min_ext.y - padding, min_ext.z - padding};
        hi = (Vec3) {max_ext.x + padding, max_ext.y + padding, max_ext.z + padding};
        // printf("domain lengths are %g by %g by %g\n", hi.x-lo.x, hi.y-lo.y, hi.z-lo.z);
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
    // ---------------------------------------------------------------------------------------
    
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
        // int cutcp_block_dim_x = cutcp_block.x;
        // int cutcp_block_dim_y = cutcp_block.y;
        // int cutcp_block_dim_z = cutcp_block.z;
        cutcp_grid.x = cutcp_blks == 0 ? cutcp_grid_dim_x * cutcp_grid_dim_y * cutcp_grid_dim_z : SM_NUM * cutcp_blks;
        cutcp_grid.y = 1;
        cutcp_grid.z = 1;
        // cutcp_block.x = cutcp_block_dim_x * cutcp_block_dim_y * cutcp_block_dim_z;
        // cutcp_block.y = 1;
        // cutcp_block.z = 1;

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

        // checkKernelErrors((mix_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_ptb_binZeroCuda, 
        //     h, cutoff2, inv_cutoff2, cutcp_ptb_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
        //     cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] cutcp took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------

    // GPTB running
    // ---------------------------------------------------------------------------------------
        cutcp_block.x = ori_cutcp_block.x * ori_cutcp_block.y * ori_cutcp_block.z;
        cutcp_block.y = 1;
        cutcp_block.z = 1;
        printf("[GPTB] Running with cutcp...\n");
        printf("[GPTB] cutcp_grid -- %d * %d * %d cutcp_block -- %d * %d * %d \n", 
            cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, cutcp_block.x, cutcp_block.y, cutcp_block.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((general_ptb_cutcp<<<cutcp_grid, cutcp_block>>>(
            binDim_x, binDim_y, cutcp_gptb_binZeroCuda, 
            h, cutoff2, inv_cutoff2, cutcp_gptb_regionZeroCuda, 25, 
            ori_cutcp_grid.x, ori_cutcp_grid.y, ori_cutcp_grid.z, ori_cutcp_block.x, ori_cutcp_block.y, ori_cutcp_block.z,
            0, cutcp_grid.x * cutcp_grid.y * cutcp_grid.z, ori_cutcp_grid.x * ori_cutcp_grid.y * ori_cutcp_grid.z, 0
            )));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[GPTB] cutcp took %f ms\n\n", kernel_time);
    // Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_cutcp_ori_regionZeroCuda, cutcp_ori_regionZeroCuda, lnall * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_cutcp_ptb_regionZeroCuda, cutcp_ptb_regionZeroCuda, lnall * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_cutcp_gptb_regionZeroCuda, cutcp_gptb_regionZeroCuda, lnall * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < lnall; i++) {
        float v1 = host_cutcp_ori_regionZeroCuda[i];
        float v2 = host_cutcp_ptb_regionZeroCuda[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
        if (i < 3) printf("%f %f\n", v1, v2);
    }
    if (errors > 0) {
        printf("ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

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

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaFree(cutcp_ori_regionZeroCuda));
    cudaErrCheck(cudaFree(cutcp_ori_binBaseCuda));
    cudaErrCheck(cudaFree(cutcp_ptb_regionZeroCuda));
    cudaErrCheck(cudaFree(cutcp_ptb_binBaseCuda));

    free(host_cutcp_ori_regionZeroCuda);
    free(host_cutcp_ptb_regionZeroCuda);

    cudaErrCheck(cudaDeviceReset());
    return 0;
}