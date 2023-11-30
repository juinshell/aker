
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

#include "header/tpacf_header.h"
#include "kernel/tpacf_kernel.cu"

#include "mix_kernel/cp-tpacf.cu" 

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
        int solo_ptb_cp_blks = 8;
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
        checkKernelErrors((mixed_cp_tpacf_kernel_6_1 <<<mix_kernel_grid, mix_kernel_block>>>(runatoms, 0.1, gptb_output, ori_cp_grid.x, ori_cp_grid.y, ori_cp_grid.z, ori_cp_block.x, ori_cp_block.y, ori_cp_block.z, 
    0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_cp_grid.x * ori_cp_grid.y * ori_cp_grid.z, tpacf_gptb_hists, tpacf_gptb_x, tpacf_gptb_y, tpacf_gptb_z, NUM_SETS, NUM_ELEMENTS, 
            ori_tpacf_grid.x, ori_tpacf_grid.y, ori_tpacf_grid.z, ori_tpacf_block.x, ori_tpacf_block.y, ori_tpacf_block.z,
            0, mix_kernel_grid.x * mix_kernel_grid.y * mix_kernel_grid.z, ori_tpacf_grid.x * ori_tpacf_grid.y * ori_tpacf_grid.z)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[MIX] cp_tpacf_6_1 took %f ms\n\n", kernel_time);
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
