#include <stdio.h>
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

// #include "header/tzgemm_ori.h"
#include "header/tzgemm_header.h"
#include "header_64/histo_header.h"

// #include "file_t/tzgemm_ori.cu"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/histo_kernel.cu"


__global__ void mix_kernel(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    uchar4 *sm_mappings,
    unsigned int num_elements,
    unsigned int sm_range_min,
    unsigned int sm_range_max,
    unsigned int histo_height,
    unsigned int histo_width,
    unsigned int *global_subhisto,
    unsigned int *global_histo,
    unsigned int *global_overflow,
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y,
    int histo_iter){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, 
        MATRIX_M, MATRIX_N, MATRIX_K,
		wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < HISTO_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_histo (
            sm_mappings,
            num_elements,
            sm_range_min,
            sm_range_max,
            histo_height,
            histo_width,
            global_subhisto,
            global_histo,
            global_overflow,
            grid_dimension_x, grid_dimension_y, block_dimension_x, block_dimension_y,
            thread_step, histo_iter);
    }
}


int main(int argc, char* argv[]) {
    int histo_blks = 1;
    int histo_iter = 3500;
	int wmma_blks = 2;
    int wmma_iter = 160;
    int M_INPUT = 16 * 8 * 16;
	int N_INPUT = 16 * 8 * 16;
	int K_INPUT = 16 * 8 * 16;
    // int M_INPUT = 128 * 1;
	// int N_INPUT = 128 * 3136;
	// int K_INPUT = 128 * 1;
	int mixwarp = 3;
    if (argc == 2) {
        mixwarp = atoi(argv[1]);
    } else if (argc == 4) {
        histo_blks = atoi(argv[1]);
        histo_iter = atoi(argv[2]);
		mixwarp = atoi(argv[3]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    float serial_time = 0;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));
    float sub_time;
    cudaEvent_t startSUBKERNEL;
    cudaEvent_t stopSUBKERNEL;
    cudaErrCheck(cudaEventCreate(&startSUBKERNEL));
    cudaErrCheck(cudaEventCreate(&stopSUBKERNEL));

	cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaErrCheck(cudaStreamCreate(&streams[i]));
    }

    // tcgemm variables
    // ---------------------------------------------------------------------------------------
        int MATRIX_M = (M_INPUT < 64) ? 64 : (M_INPUT / 64) * 64;
        int MATRIX_N = (N_INPUT < 64) ? 64 : (N_INPUT / 64) * 64;
        int MATRIX_K = (K_INPUT < 64) ? 64 : (K_INPUT / 64) * 64;

        int M_TILES = MATRIX_M / WMMA_M;
        int N_TILES = MATRIX_N / WMMA_N;
        int K_TILES = MATRIX_K / WMMA_K;

        printf("M_ORI: %5d MATRIX_M: %5d (%d x %d) \n", M_INPUT, MATRIX_M, WMMA_M, M_TILES);
        printf("N_ORI: %5d MATRIX_N: %5d (%d x %d) \n", N_INPUT, MATRIX_N, WMMA_N, N_TILES);
        printf("K_ORI: %5d MATRIX_K: %5d (%d x %d) \n", K_INPUT, MATRIX_K, WMMA_K, K_TILES);

        float *ori_host_A = NULL;
        float *ori_host_B = NULL;
        float *host_wmma_ori_c = NULL;
        float *host_wmma_ptb_c = NULL;

        half *wmma_ori_a = NULL;
        half *wmma_ori_b = NULL;
        float *wmma_ori_c = NULL;
        float *wmma_ptb_c = NULL;

        host_wmma_ori_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
        host_wmma_ptb_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);

        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * MATRIX_M * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * MATRIX_N * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_a), sizeof(half) * MATRIX_M * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_b), sizeof(half) * MATRIX_N * MATRIX_K));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_c), sizeof(float) * MATRIX_M * MATRIX_N));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ptb_c), sizeof(float) * MATRIX_M * MATRIX_N));

        assert(((unsigned long long)wmma_ori_a) % 128 == 0);
        assert(((unsigned long long)wmma_ori_b) % 128 == 0);
        assert(((unsigned long long)wmma_ori_c) % 128 == 0);
        assert(((unsigned long long)wmma_ptb_c) % 128 == 0);

        curandGenerator_t gen;
        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
        curandErrCheck(curandGenerateUniform(gen, ori_host_A, MATRIX_M * MATRIX_K));
        curandErrCheck(curandGenerateUniform(gen, ori_host_B, MATRIX_N * MATRIX_K));
        convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_a, ori_host_A, MATRIX_M * MATRIX_K);
        convertFp32ToFp16 <<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_b, ori_host_B, MATRIX_N * MATRIX_K);
        cudaErrCheck(cudaMemset(wmma_ori_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
        cudaErrCheck(cudaMemset(wmma_ptb_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
    // ---------------------------------------------------------------------------------------


    // histo variables
    // ---------------------------------------------------------------------------------------
        char inpFiles[] = "../0_mybench/header/histo_img.bin";
        unsigned int img_width, img_height;
        unsigned int histo_width, histo_height;

        FILE* f = fopen(inpFiles, "rb");
        int result = 0;
        result += fread(&img_width,    sizeof(unsigned int), 1, f);
        result += fread(&img_height,   sizeof(unsigned int), 1, f);
        result += fread(&histo_width,  sizeof(unsigned int), 1, f);
        result += fread(&histo_height, sizeof(unsigned int), 1, f);

        printf("img_height %d img_width %d \n", img_height, img_width);
        printf("histo_height %d histo_width %d \n", histo_height, histo_width);

        unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
        result = fread(img, sizeof(unsigned int), img_width*img_height, f);
        fclose(f);
        if (result != img_width*img_height){
            fputs("[Error] Reading input array from file\n", stderr);
            return -1;
        }

        int even_width = ((img_width+1)/2)*2;
        unsigned int* ori_input;
        unsigned int* ori_ranges;
        uchar4* ori_sm_mappings;
        unsigned int* ori_global_subhisto;
        unsigned short* ori_global_histo;
        unsigned int* ori_global_overflow;
        unsigned char* ori_final_histo;

        unsigned int* ptb_input;
        unsigned int* ptb_ranges;
        uchar4* ptb_sm_mappings;
        unsigned int* ptb_global_subhisto;
        unsigned short* ptb_global_histo;
        unsigned int* ptb_global_overflow;
        unsigned char* ptb_final_histo;

        unsigned char* host_ori_final_histo;
        unsigned char* host_ptb_final_histo;

        cudaMalloc((void**)&ori_input           , even_width*(((img_height+UNROLL-1)/UNROLL)*UNROLL)*sizeof(unsigned int));
        cudaMalloc((void**)&ori_ranges          , 2*sizeof(unsigned int));
        cudaMalloc((void**)&ori_sm_mappings     , img_width*img_height*sizeof(uchar4));
        cudaMalloc((void**)&ori_global_subhisto , BLOCK_X*img_width*histo_height*sizeof(unsigned int));
        cudaMalloc((void**)&ori_global_histo    , img_width*histo_height*sizeof(unsigned short));
        cudaMalloc((void**)&ori_global_overflow , img_width*histo_height*sizeof(unsigned int));
        cudaMalloc((void**)&ori_final_histo     , img_width*histo_height*sizeof(unsigned char));

        cudaMalloc((void**)&ptb_input           , even_width*(((img_height+UNROLL-1)/UNROLL)*UNROLL)*sizeof(unsigned int));
        cudaMalloc((void**)&ptb_ranges          , 2*sizeof(unsigned int));
        cudaMalloc((void**)&ptb_sm_mappings     , img_width*img_height*sizeof(uchar4));
        cudaMalloc((void**)&ptb_global_subhisto , BLOCK_X*img_width*histo_height*sizeof(unsigned int));
        cudaMalloc((void**)&ptb_global_histo    , img_width*histo_height*sizeof(unsigned short));
        cudaMalloc((void**)&ptb_global_overflow , img_width*histo_height*sizeof(unsigned int));
        cudaMalloc((void**)&ptb_final_histo , img_width*histo_height*sizeof(unsigned char));

        cudaMemset(ori_final_histo , 0 , img_width*histo_height*sizeof(unsigned char));
        cudaMemset(ptb_final_histo , 0 , img_width*histo_height*sizeof(unsigned char));

        host_ori_final_histo = (unsigned char *)calloc(histo_width*histo_height, sizeof(unsigned char));
        host_ptb_final_histo = (unsigned char *)calloc(histo_width*histo_height, sizeof(unsigned char));

        for (int y=0; y < img_height; y++){
            cudaMemcpy(&(((unsigned int*)ori_input)[y*even_width]),&img[y*img_width],img_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
            cudaMemcpy(&(((unsigned int*)ptb_input)[y*even_width]),&img[y*img_width],img_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
        }
    // ---------------------------------------------------------------------------------------


    // SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = wmma_blks == 0 ? wmma_grid_dim_x : SM_NUM * wmma_blks;
	wmma_block.x = THREADS_PER_BLOCK;

    int SHMEM_SZ = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
	cudaErrCheck(cudaFuncSetAttribute(
		ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	SHMEM_SZ = 0;

    printf("[PTB] Running with tzgemm...\n");
    printf("[PTB] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[PTB] tzgemm took %f ms\n", kernel_time);
    serial_time += kernel_time;
    // ---------------------------------------------------------------------------------------

	// SOLO running
    // ---------------------------------------------------------------------------------------
    printf("[ORI] Running with histo...\n");
    cudaErrCheck(cudaEventRecord(startKERNEL));

    unsigned int ranges_h[2] = {UINT32_MAX, 0};
    dim3 gridDim1, blockDim1;
    gridDim1.x = PRESCAN_BLOCKS_X;
    blockDim1.x = PRESCAN_THREADS;
    // printf("[ORI][histo 1] gridDim1 -- %d * %d blockDim1 -- %d * %d \n", 
    //     gridDim1.x, gridDim1.y, blockDim1.x, blockDim1.y);
    cudaMemcpy(ori_ranges, ranges_h, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
    checkKernelErrors((histo_prescan_kernel<<<gridDim1, blockDim1>>>(
            (unsigned int*)ori_input, img_height*img_width, ori_ranges)));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    // printf("[ORI] sub took %f ms\n", sub_time);
    
    dim3 gridDim2, blockDim2;
    gridDim2.x = (img_height + UNROLL-1)/UNROLL;
    blockDim2.x = (img_width+1)/2;
    // printf("[ORI][histo 2] gridDim2 -- %d * %d blockDim2 -- %d * %d \n", gridDim2.x, gridDim2.y, blockDim2.x, blockDim2.y);
    cudaMemcpy(ranges_h,ori_ranges, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemset(ori_global_subhisto,0,img_width*histo_height*sizeof(unsigned int));

    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
    checkKernelErrors((histo_intermediates_kernel<<<gridDim2, blockDim2>>>(
            (uint2*)(ori_input),
            (unsigned int)img_height,
            (unsigned int)img_width,
            (img_width+1)/2,
            (uchar4*)(ori_sm_mappings))));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    // printf("[ORI] sub took %f ms\n", sub_time);

    dim3 gridDim3, blockDim3;
    gridDim3.x = BLOCK_X;
    gridDim3.y = ranges_h[1]-ranges_h[0]+1;
    blockDim3.x = THREADS;
    printf("[ORI] gridDim3 -- %d * %d blockDim3 -- %d * %d \n", gridDim3.x, gridDim3.y, blockDim3.x, blockDim3.y);
    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
    checkKernelErrors((ori_histo<<<gridDim3, blockDim3>>>(
            (uchar4*)(ori_sm_mappings),
            img_height*img_width,
            ranges_h[0], ranges_h[1],
            histo_height, histo_width,
            (unsigned int*)(ori_global_subhisto),
            (unsigned int*)(ori_global_histo),
            (unsigned int*)(ori_global_overflow),
            histo_iter
            )));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    printf("[ORI] histo took %f ms\n\n", sub_time);

    serial_time += sub_time;

    dim3 gridDim4, blockDim4;
    gridDim4.x = BLOCK_X * 3;
    blockDim4.x = 512;
    // printf("[ORI][histo 4] gridDim4 -- %d * %d blockDim4 -- %d * %d \n", gridDim4.x, gridDim4.y, blockDim4.x, blockDim4.y);
    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
    checkKernelErrors((histo_final_kernel<<<gridDim4, blockDim4>>>(
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(ori_global_subhisto),
                (unsigned int*)(ori_global_histo),
                (unsigned int*)(ori_global_overflow),
                (unsigned int*)(ori_final_histo))));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    // printf("[ORI] sub took %f ms\n", sub_time);

    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    // printf("[ORI] histo all took %f ms\n\n", kernel_time);

    // PTB running
    // ---------------------------------------------------------------------------------------
    // printf("[PTB] Running with histo...\n");
    cudaErrCheck(cudaEventRecord(startKERNEL));
    
    ranges_h[0] = UINT32_MAX; ranges_h[1] = 0;
    // dim3 gridDim1, blockDim1;
    gridDim1.x = PRESCAN_BLOCKS_X;
    blockDim1.x = PRESCAN_THREADS;
    cudaMemcpy(ptb_ranges,ranges_h, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);
    checkKernelErrors((histo_prescan_kernel<<<gridDim1, blockDim1>>>(
            (unsigned int*)ptb_input, img_height*img_width, ptb_ranges
            // , grid1_dimension_x, grid1_dimension_y, block1_dimension_x, block1_dimension_y
            )));
    
    // dim3 gridDim2, blockDim2;
    gridDim2.x = (img_height + UNROLL-1)/UNROLL;
    blockDim2.x = (img_width+1)/2;
    cudaMemcpy(ranges_h,ptb_ranges, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemset(ptb_global_subhisto,0,img_width*histo_height*sizeof(unsigned int));
    checkKernelErrors((histo_intermediates_kernel<<<gridDim2, blockDim2>>>(
            (uint2*)(ptb_input),
            (unsigned int)img_height,
            (unsigned int)img_width,
            (img_width+1)/2,
            (uchar4*)(ptb_sm_mappings)
            //, grid2_dimension_x, grid2_dimension_y, block2_dimension_x, block2_dimension_y
            )));

    // dim3 gridDim3, blockDim3;
    gridDim3.x = BLOCK_X;
    gridDim3.y = ranges_h[1]-ranges_h[0]+1;
    blockDim3.x = THREADS;
    int grid3_dimension_x = gridDim3.x;
    int grid3_dimension_y = gridDim3.y;
    int block3_dimension_x = blockDim3.x;
    int block3_dimension_y = blockDim3.y;
    gridDim3.x = histo_blks == 0 ? grid3_dimension_x * grid3_dimension_y : 68 * histo_blks;
    gridDim3.y = 1;
    blockDim3.x = block3_dimension_x * block3_dimension_y;
    blockDim3.y = 1;

	if (mixwarp == 1) {
		dim3 mix_grid, mix_block;
		mix_grid.x = (gridDim3.x > wmma_grid.x) ? gridDim3.x : wmma_grid.x;
        mix_grid.y = 1;
        mix_block.x = blockDim3.x + wmma_block.x;
        mix_block.y = 1;
        printf("[PTB] gridDim3 -- %d * %d blockDim3 -- %d * %d \n", gridDim3.x, gridDim3.y, blockDim3.x, blockDim3.y);
		printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// histo parameters
			(uchar4*)(ori_sm_mappings),
            img_height*img_width,
            ranges_h[0], ranges_h[1],
            histo_height, histo_width,
            (unsigned int*)(ptb_global_subhisto),
            (unsigned int*)(ptb_global_histo),
            (unsigned int*)(ptb_global_overflow),
            grid3_dimension_x, grid3_dimension_y, block3_dimension_x, block3_dimension_y,
            histo_iter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 2) {
		cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
        checkKernelErrors((ptb_histo<<<gridDim3, blockDim3, 0, streams[1]>>>(
            (uchar4*)(ori_sm_mappings),
            img_height*img_width,
            ranges_h[0], ranges_h[1],
            histo_height, histo_width,
            (unsigned int*)(ptb_global_subhisto),
            (unsigned int*)(ptb_global_histo),
            (unsigned int*)(ptb_global_overflow),
            grid3_dimension_x, grid3_dimension_y, block3_dimension_x, block3_dimension_y,
            histo_iter
            )));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] gridDim3 -- %d * %d blockDim3 -- %d * %d \n", gridDim3.x, gridDim3.y, blockDim3.x, blockDim3.y);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
        gridDim3.x = BLOCK_X;
        gridDim3.y = ranges_h[1]-ranges_h[0]+1;
        blockDim3.x = THREADS;

        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_histo<<<gridDim3, blockDim3, 0, streams[1]>>>(
            (uchar4*)(ori_sm_mappings),
            img_height*img_width,
            ranges_h[0], ranges_h[1],
            histo_height, histo_width,
            (unsigned int*)(ptb_global_subhisto),
            (unsigned int*)(ptb_global_histo),
            (unsigned int*)(ptb_global_overflow),
            histo_iter
            )));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[STREAMO] mix took %f ms\n\n", kernel_time);
    }

    // dim3 gridDim4, blockDim4;
    gridDim4.x = BLOCK_X * 3;
    blockDim4.x = 512;
    checkKernelErrors((histo_final_kernel<<<gridDim4, blockDim4>>>(
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(ptb_global_subhisto),
                (unsigned int*)(ptb_global_histo),
                (unsigned int*)(ptb_global_overflow),
                (unsigned int*)(ptb_final_histo)
                )));
    
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // ---------------------------------------------------------------------------------------

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

	// Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_wmma_ori_c, wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_wmma_ptb_c, wmma_ptb_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_ori_final_histo, ori_final_histo, histo_height*histo_width*sizeof(unsigned char), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_ptb_final_histo, ptb_final_histo, histo_height*histo_width*sizeof(unsigned char), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = host_wmma_ori_c[i];
        float v2 = host_wmma_ptb_c[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[WMMA] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[WMMA] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }
    errors = 0;
    for (int i = 0; i < histo_height*histo_width; i++) {
        unsigned int v1 = (unsigned int)host_ori_final_histo[i];
        unsigned int v2 = (unsigned int)host_ptb_final_histo[i];
        if (v1 != v2) {
            errors++;
            if (errors < 10) printf("%d %d\n", v1, v2);
        }
        if (i < 3) printf("%d %d\n", v1, v2);
    }
    if (errors > 0) {
        printf("ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}