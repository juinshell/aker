#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>
#include <assert.h>
#include <curand.h>
#include <cublas_v2.h>

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


#include "header/tzgemm_header.h"
#include "header/mrif_header.h"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/mrif_kernel.cu"


__global__ void mix_kernel0 (
    half *a, half *b, float *c,
	int matrixM, int matrixN, int matrixK,
    int wmmaGridX, int wmmaBlockX, 
    int wmmaIter,
    int numK, int kGlobalIndex,
	float* x, float* y, float* z, 
	float* outR, float* outI, 
	int mrif_grid_dim_x, int mrif_block_dim_x,
	int mrifIter){
    if (threadIdx.x < wmmaBlockX * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, 
            matrixM, matrixN, matrixK,
            wmmaGridX, wmmaBlockX, wmmaIter);
    } else if (threadIdx.x >= wmmaBlockX * 1 && blockIdx.x < MRIF_GRID_DIM) {
        int thread_step = wmmaBlockX * 1;
        mix_mrif(numK, kGlobalIndex,
              x, y, z, 
              outR, outI, 
			  mrif_grid_dim_x, mrif_block_dim_x, thread_step,
			  mrifIter);
    }
}

__global__ void mix_kernel1 (
    half *a, half *b, float *c,
	int matrixM, int matrixN, int matrixK,
    int wmmaGridX, int wmmaBlockX, 
    int wmmaIter,
    int numK, int kGlobalIndex,
	float* x, float* y, float* z, 
	float* outR, float* outI, 
	int mrif_grid_dim_x, int mrif_block_dim_x,
	int mrifIter){
    int thread_step = wmmaBlockX * 1;
    if (threadIdx.x < wmmaBlockX) {
        mix_tzgemm0(a, b, c, 
            matrixM, matrixN, matrixK,
            wmmaGridX, wmmaBlockX, wmmaIter);
    } else if (threadIdx.x < wmmaBlockX * 2) {
        thread_step = wmmaBlockX;
        mix_tzgemm1(a, b, c, 
            matrixM, matrixN, matrixK,
            wmmaGridX, wmmaBlockX, wmmaIter, thread_step);
    } else {
        thread_step = wmmaBlockX * 2;
        mix_mrif(numK, kGlobalIndex,
              x, y, z, 
              outR, outI, 
			  mrif_grid_dim_x, mrif_block_dim_x, thread_step,
			  mrifIter);
    }
}


int main(int argc, char* argv[]) {
    int mrifBlks = 1;
    int mrifIter = 1;
	int wmmaBlks = 2;
    int wmmaIter = 1;
    int inputM = 128 * 1;
	int inputN = 128 * 3136;
	int inputK = 128 * 1;
	int coloCase = 1;
	if (argc == 2) {
		coloCase = atoi(argv[1]);
	} else if (argc == 4) {
        mrifBlks = atoi(argv[1]);
        mrifIter = atoi(argv[2]);
		coloCase = atoi(argv[3]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
        float kernelTime = 0;
        float serialTime = 0;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
        cudaStream_t streams[2];
        for (int i = 0; i < 2; i++) {
            cudaErrCheck(cudaStreamCreate(&streams[i]));
        }
    // ---------------------------------------------------------------------------------------

    // tzgemm variables
    // ---------------------------------------------------------------------------------------
        int matrixM = (inputM < 64) ? 64 : (inputM / 64) * 64;
        int matrixN = (inputN < 64) ? 64 : (inputN / 64) * 64;
        int matrixK = (inputK < 64) ? 64 : (inputK / 64) * 64;
        int tileM = matrixM / WMMA_M;
        int tileN = matrixN / WMMA_N;
        int tileK = matrixK / WMMA_K;
        printf("M_ORI: %5d matrixM: %5d (%d x %d) \n", inputM, matrixM, WMMA_M, tileM);
        printf("N_ORI: %5d matrixN: %5d (%d x %d) \n", inputN, matrixN, WMMA_N, tileN);
        printf("K_ORI: %5d matrixK: %5d (%d x %d) \n", inputK, matrixK, WMMA_K, tileK);

        float *oriDevA = NULL;
        float *oriDevB = NULL;
        float *oriHostC = NULL;
        float *ptbHostC = NULL;

        half *soloA = NULL;
        half *soloB = NULL;
        float *soloC = NULL;
        half *coloA = NULL;
        half *coloB = NULL;
        float *coloC = NULL;

        oriHostC = (float *)malloc(sizeof(float) * matrixM * matrixN);
        ptbHostC = (float *)malloc(sizeof(float) * matrixM * matrixN);
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&oriDevA), sizeof(float) * matrixM * matrixK));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&oriDevB), sizeof(float) * matrixN * matrixK));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&soloA), sizeof(half) * matrixM * matrixK));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&soloB), sizeof(half) * matrixN * matrixK));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&soloC), sizeof(float) * matrixM * matrixN));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&coloA), sizeof(half) * matrixM * matrixK));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&coloB), sizeof(half) * matrixN * matrixK));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&coloC), sizeof(float) * matrixM * matrixN));

        assert(((unsigned long long)soloA) % 128 == 0);
        assert(((unsigned long long)soloB) % 128 == 0);
        assert(((unsigned long long)soloC) % 128 == 0);
        assert(((unsigned long long)coloA) % 128 == 0);
        assert(((unsigned long long)coloB) % 128 == 0);
        assert(((unsigned long long)coloC) % 128 == 0);

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(gen, oriDevA, matrixM * matrixK));
        curandErrCheck(curandGenerateUniform(gen, oriDevB, matrixN * matrixK));
        convertFp32ToFp16 <<< (matrixM * matrixK + 255) / 256, 256 >>> (soloA, oriDevA, matrixM * matrixK);
        convertFp32ToFp16 <<< (matrixN * matrixK + 255) / 256, 256 >>> (soloB, oriDevB, matrixN * matrixK);
        cudaErrCheck(cudaMemcpy(coloA, soloA, matrixM * matrixK * sizeof(half), cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(coloB, soloB, matrixN * matrixK * sizeof(half), cudaMemcpyDeviceToDevice));
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

        // kernel 1
        float *soloPhiR, *soloPhiI;
        float *soloR, *soloI;
        float *soloRealRhoPhi, *soloImagRhoPhi;
        // kernel 2
        float *soloX, *soloY, *soloZ;
        float *soloOutI, *soloOutR;
        float *hostSoloI;		            /* Output signal (complex) */
        // kernel 2
        float *coloX, *coloY, *coloZ;
        float *coloOutI, *coloOutR;
        float *hostColoI;		            /* Output signal (complex) */

        cudaErrCheck(cudaMalloc((void **)&soloPhiR, numK * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&soloPhiI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&soloR, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&soloI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&soloRealRhoPhi, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&soloImagRhoPhi, numK * sizeof(float)));
        // host_ori_phiMag = (float* ) memalign(16, numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(soloPhiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(soloPhiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(soloR, base_dR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(soloI, base_dI, numK * sizeof(float), cudaMemcpyHostToDevice));

        base_realRhoPhi = (float* ) calloc(numK, sizeof(float));
        base_imagRhoPhi = (float* ) calloc(numK, sizeof(float));

        cudaErrCheck(cudaMalloc((void **)&soloX, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&soloY, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&soloZ, numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(soloX, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(soloY, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(soloZ, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&soloOutR, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&soloOutI, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(soloOutR, 0, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(soloOutI, 0, numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&coloX, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&coloY, numX * sizeof(float)));   
        cudaErrCheck(cudaMalloc((void **)&coloZ, numX * sizeof(float)));   
        cudaErrCheck(cudaMemcpy(coloX, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(coloY, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(coloZ, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&coloOutR, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&coloOutI, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(coloOutR, 0, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(coloOutI, 0, numX * sizeof(float)));

        hostSoloI = (float*) calloc (numX, sizeof (float));
        hostColoI = (float*) calloc (numX, sizeof (float));
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 wmmaGrid;
        dim3 wmmaBlock;
        wmmaGrid.x = (tileM * tileN) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
        wmmaBlock.x = THREADS_PER_BLOCK;

        int wmmaGridX = (tileM * tileN) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
        int wmmaBlockX = wmmaBlock.x;
        wmmaGrid.x = wmmaBlks == 0 ? wmmaGridX : SM_NUM * wmmaBlks;
        wmmaBlock.x = THREADS_PER_BLOCK;

        int shareMemSize = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * 
                (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
        if (wmmaBlks != 0) {
            shareMemSize = 0;
        } else {
            cudaErrCheck(cudaFuncSetAttribute(ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, shareMemSize));
        }

        printf("[PTB] Running with tzgemm...\n");
        printf("[PTB] wmmaGrid -- %d * %d wmmaBlock -- %d * %d \n", wmmaGrid.x, wmmaGrid.y, wmmaBlock.x, wmmaBlock.y);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_tzgemm<<<wmmaGrid, wmmaBlock, shareMemSize, streams[0]>>>(
                soloA, soloB, soloC, matrixM, matrixN, matrixK, wmmaGridX, wmmaBlockX, wmmaIter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernelTime, startKERNEL, stopKERNEL));
        printf("[PTB] tzgemm took %f ms\n", kernelTime);
        serialTime += kernelTime;
    // ---------------------------------------------------------------------------------------


	// mrif kernel 1
    // ---------------------------------------------------------------------------------------
        dim3 mrifGrid1;
        dim3 mrifBlock1;
        mrifGrid1.x = numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrifGrid1.y = 1;
        mrifBlock1.x = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
        mrifBlock1.y = 1;
        printf("[ORI] mrifGrid1 -- %d * %d * %d mrifBlock1 -- %d * %d * %d \n", 
                mrifGrid1.x, mrifGrid1.y, mrifGrid1.z, mrifBlock1.x, mrifBlock1.y, mrifBlock1.z);
        checkKernelErrors((ComputeRhoPhiGPU <<< mrifGrid1, mrifBlock1 >>> (
                numK, soloPhiR, soloPhiI, soloR, soloI, soloRealRhoPhi, soloImagRhoPhi)));
        cudaErrCheck(cudaMemcpy(base_realRhoPhi, soloRealRhoPhi, numK * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(base_imagRhoPhi, soloImagRhoPhi, numK * sizeof(float), cudaMemcpyDeviceToHost));

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
        // int FHGrids = numK / KERNEL_FH_K_ELEMS_PER_GRID;
        dim3 mrifGrid2;
        dim3 mrifBlock2;
        mrifGrid2.x = numX / KERNEL_FH_THREADS_PER_BLOCK;
        mrifGrid2.y = 1;
        mrifBlock2.x = KERNEL_FH_THREADS_PER_BLOCK;
        mrifBlock2.y = 1;
        printf("[ORI] mrifGrid2 -- %d * %d * %d mrifBlock2 -- %d * %d * %d \n", 
                mrifGrid2.x, mrifGrid2.y, mrifGrid2.z, mrifBlock2.x, mrifBlock2.y, mrifBlock2.z);
        // printf("FHGrids %d \n", FHGrids);

        int FHGridBase = 0 * KERNEL_FH_K_ELEMS_PER_GRID;
        kValues* kValsTile = kVals + FHGridBase;
        int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, numK - FHGridBase);
        cudaMemcpyToSymbol(c, kValsTile, numElems * sizeof(kValues), 0);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_mrif <<< mrifGrid2, mrifBlock2 >>> (
                numK, FHGridBase, soloX, soloY, soloZ, soloOutR, soloOutI, mrifIter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernelTime, startKERNEL, stopKERNEL));
        printf("[ORI] mrif took %f ms\n\n", kernelTime);
        serialTime += kernelTime;
    // ---------------------------------------------------------------------------------------


	// MIX running 
    // ---------------------------------------------------------------------------------------
	if (coloCase == 1) {
        int mrifGridX2 = mrifGrid2.x;
        int mrifBlockX2 = mrifBlock2.x;
        mrifGrid2.x = mrifBlks == 0 ? mrifGridX2 : SM_NUM * mrifBlks;

		dim3 mixGrid, mixBlock;
        mixGrid.x = (mrifGrid2.x > wmmaGrid.x) ? mrifGrid2.x : wmmaGrid.x;
        mixGrid.y = 1;
        mixBlock.x = mrifBlock2.x + wmmaBlock.x;
        mixBlock.y = 1;

        mixGrid.x = SM_NUM;
        mixBlock.x = 512;

        printf("[PTB] mrifGrid2 -- %d * %d * %d mrifBlock2 -- %d * %d * %d \n", 
            mrifGrid2.x, mrifGrid2.y, mrifGrid2.z, mrifBlock2.x, mrifBlock2.y, mrifBlock2.z);
		printf("[MIX] mixGrid -- %d * %d mixBlock -- %d * %d \n", mixGrid.x, mixGrid.y, mixBlock.x, mixBlock.y);


		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel1 <<<mixGrid, mixBlock>>> (
			// wmma parameters
			coloA, coloB, coloC, 
			matrixM, matrixN, matrixK,
			wmmaGridX, wmmaBlockX, wmmaIter,
			// mrif parameters
			numK, FHGridBase, 
            coloX, coloY, coloZ, coloOutR, coloOutI, 
            mrifGridX2, mrifBlockX2,
            mrifIter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernelTime, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernelTime);
	} else if (coloCase == 2) {
        int mrifGridX2 = mrifGrid2.x;
        int mrifBlockX2 = mrifBlock2.x;
        mrifGrid2.x = mrifBlks == 0 ? mrifGridX2 : SM_NUM * mrifBlks;

		cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_tzgemm<<<wmmaGrid, wmmaBlock, shareMemSize, streams[0]>>>(coloA, coloB, coloC, 
							matrixM, matrixN, matrixK,
							// alpha, beta,
							wmmaGridX, wmmaBlockX, wmmaIter)));
        checkKernelErrors((ptb_mrif <<< mrifGrid2, mrifBlock2, 0, streams[1] >>> (numK, FHGridBase, 
                            coloX, coloY, coloZ, coloOutR, coloOutI, 
                            mrifGridX2, mrifBlockX2,
                            mrifIter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernelTime, startKERNEL, stopKERNEL));
        printf("[PTB] mrifGrid2 -- %d * %d * %d mrifBlock2 -- %d * %d * %d \n", 
            mrifGrid2.x, mrifGrid2.y, mrifGrid2.z, mrifBlock2.x, mrifBlock2.y, mrifBlock2.z);
		printf("[STREAMP] mix took %f ms\n\n", kernelTime);
	} else if (coloCase == 3) {
        cudaErrCheck(cudaEventRecord(startKERNEL));
        int32_t index = 0;
        while (index++ < 0) {
            checkKernelErrors((ptb_tzgemm<<<wmmaGrid, wmmaBlock, shareMemSize, streams[0]>>>(
                    coloA, coloB, coloC, matrixM, matrixN, matrixK, wmmaGridX, wmmaBlockX, wmmaIter)));
            checkKernelErrors((ori_mrif <<< mrifGrid2, mrifBlock2, 0, streams[1] >>> (numK, FHGridBase, 
                    coloX, coloY, coloZ, coloOutR, coloOutI, mrifIter)));
        }
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernelTime, startKERNEL, stopKERNEL));
		printf("[STREAMO] mix took %f ms\n\n", kernelTime);
    }

    printf("[STAT] Overlap rate: %.2f\n", (serialTime - kernelTime) * 100 / serialTime);
    printf("[STAT] Throughput speedup: %.2f\n", (serialTime / kernelTime - 1) * 100);

	// Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(oriHostC, soloC, matrixM * matrixN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(ptbHostC, coloC, matrixM * matrixN * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(hostSoloI, soloOutI, numX * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(hostColoI, coloOutI, numX * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < matrixM * matrixN; i++) {
        float v1 = oriHostC[i];
        float v2 = ptbHostC[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[wmma] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[wmma] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }
    errors = 0;
    for (int i = 0; i < numX; i++) {
        float v1 = hostSoloI[i];
        float v2 = hostColoI[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 5) printf("%f %f\n", v1, v2);
        }
        if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[mrif] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[mrif] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}