
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

#include "header/cp_header.h"
#include "kernel/cp_kernel.cu"

#include <mma.h>
using namespace nvcuda; 
#include "header/atom.h"
#include "header/cutcp_header.h"
#include "kernel/cutcp_kernel.cu"

#include "mix_kernel/cp-cutcp.cu" 

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

    // cp variables
    // ---------------------------------------------------------------------------------------
		int cp_blks = 8;
	    int cp_iter = 1;
        float *atoms = NULL;
		int atomcount = ATOMCOUNT;
		const float gridspacing = 0.1;					// number of atoms to simulate
		dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
		initatoms(&atoms, atomcount, volsize, gridspacing);

		// allocate and initialize the GPU output array
		int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

        float *ori_output;	
		float *ptb_output;
		float *gptb_output;
		cudaErrCheck(cudaMalloc((void**)&ori_output, volmemsz));
		cudaErrCheck(cudaMemset(ori_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
		cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
		cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
		cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
		float *host_ori_energy = (float *) malloc(volmemsz);
		float *host_ptb_energy = (float *) malloc(volmemsz);
		float *host_gptb_energy = (float *) malloc(volmemsz);

        dim3 cp_grid, cp_block, ori_cp_grid, ori_cp_block;
        int atomstart = 1;
		int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
		cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
		cp_block.y = BLOCKSIZEY;
		cp_block.z = 1;
		cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
		cp_grid.y = volsize.y / cp_block.y; 
		cp_grid.z = volsize.z / cp_block.z; 
		ori_cp_grid = cp_grid;
		ori_cp_block = cp_block;
		printf("[ORI] Running with cp...\n");
		printf("[ORI] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ori_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ori_output, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[ORI] cp took %f ms\n\n", kernel_time);

        cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

	// PTB running
    // ---------------------------------------------------------------------------------------
        int solo_ptb_cp_blks = 6;
	    cp_iter = 1;
		int cp_grid_dim_x = cp_grid.x;
		int cp_grid_dim_y = cp_grid.y;
		cp_grid.x = solo_ptb_cp_blks == 0 ? cp_grid_dim_x * cp_grid_dim_y : SM_NUM * solo_ptb_cp_blks;
		cp_grid.y = 1;
		printf("[PTB] Running with cp...\n");
		printf("[PTB] cp_grid -- %d * %d * %d cp_block -- %d * %d * %d\n", 
					cp_grid.x, cp_grid.y, cp_grid.z, cp_block.x, cp_block.y, cp_block.z);

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb2_cp<<<cp_grid, cp_block, 0>>>(runatoms, 0.1, ptb_output, 
			cp_grid_dim_x, cp_grid_dim_y, cp_iter)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PTB] cp took %f ms\n\n", kernel_time);

        cudaMemcpy(host_ptb_energy, ptb_output, volmemsz,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // ---------------------------------------------------------------------------------------

		atomstart = 1;
		runatoms = MAXATOMS;
		copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

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


  // MIX
  // ---------------------------------------------------------------------------------------
        dim3 mix_kernel_grid = dim3(136, 1, 1);
        dim3 mix_kernel_block = dim3(512, 1, 1);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((mixed_cp_cutcp_kernel_1_3 <<<mix_kernel_grid, mix_kernel_block>>>(runatoms, 0.1, gptb_output, ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z, 
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cp_grid.x * ori_cp_grid.y * ori_cp_grid.z, binDim_x, binDim_y, cutcp_gptb_binZeroCuda,
    h, cutoff2, inv_cutoff2, cutcp_gptb_regionZeroCuda, 25, 
    ori_cutcp_grid.x, ori_cutcp_grid.y, ori_cutcp_grid.z, ori_cutcp_block.x, ori_cutcp_block.y, ori_cutcp_block.z,
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cutcp_grid.x * ori_cutcp_grid.y * ori_cutcp_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cp_cutcp 1_3 took %f ms\n\n", kernel_time);
  // ---------------------------------------------------------------------------------------


	// Checking results
    // ---------------------------------------------------------------------------------------
    	cudaMemcpy(host_ori_energy, ori_output, volmemsz,  cudaMemcpyDeviceToHost);
	    cudaMemcpy(host_gptb_energy, gptb_output, volmemsz,  cudaMemcpyDeviceToHost);
            
        errors = 0;
        for (int i = 0; i < volsize.x * volsize.y * volsize.z; i++) {
            float v1 = host_ori_energy[i];
            float v2 = host_gptb_energy[i];
            if (fabs(v1 - v2) > 0.001f) {
                errors++;
                if (errors < 10) printf("%f %f\n", v1, v2);
            }
        }
        if (errors > 0) {
            printf("ORI VERSION does not agree with GPTB VERSION! %d errors!\n", errors);
        }
        else {
            printf("Results verified: ORIG VERSION and GPTB VERSION agree.\n");
        }
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


}
