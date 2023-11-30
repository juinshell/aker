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
#include "header/spmv_header.h"
#include "file_t/spmv_t/convert_dataset.h"
#include "file_t/spmv_t/mmio.h"

// #include "file_t/tzgemm_ori.cu"
#include "file_t/tzgemm_kernel.cu"
#include "file_t/spmv_t/spmv_kernel.cu"


__global__ void mix_kernel0(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    float *dst_vector,
    const float *d_data,const int *d_index, const int *d_perm,
    const float *x_vec,const int *d_nzcnt, const int dim,
    int spmv_grid_dim_x, int spmv_grid_dim_y, int spmv_block_dim_x, int spmv_block_dim_y,
    int spmv_iter){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, 
            MATRIX_M, MATRIX_N, MATRIX_K,
            wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < SPMV_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_spmv(dst_vector,
            d_data, d_index,  d_perm,
            x_vec, d_nzcnt,  dim,
            spmv_grid_dim_x, spmv_grid_dim_y, spmv_block_dim_x, spmv_block_dim_y,
            thread_step, spmv_iter);
    }
}


int main(int argc, char* argv[]) {
    int spmv_blks = 2;
    int spmv_iter = 17000;
	int wmma_blks = 2;
    int wmma_iter = 1900;
    int M_INPUT = 128 * 1;
	int N_INPUT = 128 * 3136;
	int K_INPUT = 128 * 1;
	int mixwarp = 2;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        spmv_blks = atoi(argv[1]);
        spmv_iter = atoi(argv[2]);
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

    // spmv variables
    // ---------------------------------------------------------------------------------------

        //parameters declaration
        int len;
        int depth;
        int dim;
        int pad=32;
        int nzcnt_len;
        
        //host memory allocation
        //matrix
        float *h_data;
        int *h_indices;
        int *h_ptr;
        int *h_perm;
        int *h_nzcnt;
        //vector
        float *host_spmv_ori_result_vector;
        float *host_spmv_ptb_result_vector;
        float *host_ori_vector;

        char mtx_bin[] = "../0_mybench/file_t/spmv_t/spmv_mtx.bin";
        char vec_bin[] = "../0_mybench/file_t/spmv_t/spmv_vec.bin";

        //device memory allocation
        //matrix
        float *spmv_ori_matrix;
        int *spmv_ori_matrix_indice;
        int *spmv_ori_matrix_perm;
        int *spmv_ori_matrix_nzcnt;
        //vector
        float *spmv_ori_result_vector;
        float *spmv_ori_vector;
        //matrix
        float *spmv_ptb_matrix;
        int *spmv_ptb_matrix_indice;
        int *spmv_ptb_matrix_perm;
        int *spmv_ptb_matrix_nzcnt;
        //vector
        float *spmv_ptb_result_vector;
        float *spmv_ptb_vector;

        int col_count;
        coo_to_jds(
            mtx_bin, // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
            1, // row padding
            pad, // warp size, IMPORTANT: change in kernel as well
            1, // pack size
            1, // is mirrored?
            0, // binary matrix
            1, // debug level [0:2]
            &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
            &col_count, &dim, &len, &nzcnt_len, &depth
        );

        host_spmv_ori_result_vector = (float*)malloc(sizeof(float)*dim); 
        host_spmv_ptb_result_vector = (float*)malloc(sizeof(float)*dim); 
        host_ori_vector = (float*)malloc(sizeof(float)*dim);
        input_vec(vec_bin, host_ori_vector,dim);

        // int len = 3637664;
        // int depth = 50;
        // int dim = 146689;
        // int nzcnt_len = 4585;
        // int pad = 32;

        cudaErrCheck(cudaMalloc((void **)&spmv_ori_matrix, len*sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ori_matrix_indice, len*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ori_matrix_perm, dim*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ori_matrix_nzcnt, nzcnt_len*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ori_vector, dim*sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ori_result_vector,dim*sizeof(float)));
        cudaErrCheck(cudaMemset((void *)spmv_ori_result_vector, 0, dim*sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_matrix, len*sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_matrix_indice, len*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_matrix_perm, dim*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_matrix_nzcnt, nzcnt_len*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_vector, dim*sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_result_vector,dim*sizeof(float)));
        cudaErrCheck(cudaMemset((void *)spmv_ptb_result_vector, 0, dim*sizeof(float)));

        //memory copy
        cudaErrCheck(cudaMemcpy(spmv_ori_matrix, h_data, len*sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(spmv_ori_matrix_indice, h_indices, len*sizeof(int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(spmv_ori_matrix_perm, h_perm, dim*sizeof(int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(spmv_ori_vector, host_ori_vector, dim*sizeof(int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth*sizeof(int)));
        cudaErrCheck(cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt,nzcnt_len*sizeof(int)));
        cudaErrCheck(cudaMemcpy(spmv_ptb_matrix, h_data, len*sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(spmv_ptb_matrix_indice, h_indices, len*sizeof(int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(spmv_ptb_matrix_perm, h_perm, dim*sizeof(int), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(spmv_ptb_vector, host_ori_vector, dim*sizeof(int), cudaMemcpyHostToDevice));
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
	if (wmma_blks != 0) {
        SHMEM_SZ = 0;
    }

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

	// SOLO running
    // ---------------------------------------------------------------------------------------
    int _thread;
    int _block;
    if (nzcnt_len * pad > 68 * 1024){
        _thread = 1024 / 16;
		_block = (nzcnt_len*pad+_thread-1)/_thread;

        // _thread = max_thread/max_block;
		// _grid = (nzcnt_len*pad+_thread-1)/_thread;
    }
    printf("_thread %d _block %d \n", _thread, _block);

    dim3 spmv_grid;
    dim3 spmv_block;
    spmv_grid.x = 1147;
    spmv_grid.y = spmv_grid.z = 1;
    spmv_block.x = 128;
    spmv_block.y = spmv_block.z = 1;

    // spmv_grid.x = _block;
    // spmv_grid.y = spmv_grid.z = 1;
    // spmv_block.x = _thread;
    // spmv_block.y = spmv_block.z = 1;

    printf("[ORI] Running with spmv...\n");
    printf("[ORI] spmv_grid -- %d * %d * %d spmv_block -- %d * %d * %d \n", 
        spmv_grid.x, spmv_grid.y, spmv_grid.z, spmv_block.x, spmv_block.y, spmv_block.z);
    
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ori_spmv<<<spmv_grid, spmv_block>>> (spmv_ori_result_vector, spmv_ori_matrix, 
        spmv_ori_matrix_indice, spmv_ori_matrix_perm, 
        spmv_ori_vector, spmv_ori_matrix_nzcnt, dim, spmv_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] spmv took %f ms\n\n", kernel_time);

    // PTB
    // ---------------------------------------------------------------------------------------
    int spmv_grid_dim_x = spmv_grid.x;
    int spmv_grid_dim_y = spmv_grid.y;
    int spmv_block_dim_x = spmv_block.x;
    int spmv_block_dim_y = spmv_block.y;
    spmv_grid.x = spmv_blks == 0 ? spmv_grid_dim_x * spmv_grid_dim_y : 68 * spmv_blks;
    spmv_grid.y = 1;
    spmv_block.x = spmv_block_dim_x * spmv_block_dim_y;
    spmv_block.y = 1;
    
    cudaErrCheck(cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth*sizeof(int)));
	cudaErrCheck(cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt,nzcnt_len*sizeof(int)));

    // printf("[PTB] Running with spmv...\n");
    // printf("[PTB] spmv_grid -- %d * %d * %d spmv_block -- %d * %d * %d \n", 
    //     spmv_grid.x, spmv_grid.y, spmv_grid.z, spmv_block.x, spmv_block.y, spmv_block.z);
    // cudaErrCheck(cudaEventRecord(startKERNEL));
    // checkKernelErrors((ptb_spmv<<<spmv_grid, spmv_block>>> (spmv_ptb_result_vector, spmv_ptb_matrix, 
    //                 spmv_ptb_matrix_indice, spmv_ptb_matrix_perm, 
    //                 spmv_ptb_vector, spmv_ptb_matrix_nzcnt, dim,
    //                 spmv_grid_dim_x, spmv_grid_dim_y, spmv_block_dim_x, spmv_block_dim_y, spmv_iter)));
    // cudaErrCheck(cudaEventRecord(stopKERNEL));
    // cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    // cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    // printf("[PTB] spmv took %f ms\n\n", kernel_time);
    serial_time += kernel_time;

	// MIX running 
    // ----------------------------------------------------------------------------------------------------------------------
	if (mixwarp == 1) {
		dim3 mix_grid, mix_block;
        mix_grid.x = (spmv_grid.x > wmma_grid.x) ? spmv_grid.x : wmma_grid.x;
        mix_grid.y = 1;
        mix_block.x = spmv_block.x + wmma_block.x;
        mix_block.y = 1;
        printf("[PTB] spmv_grid -- %d * %d * %d spmv_block -- %d * %d * %d \n", 
        spmv_grid.x, spmv_grid.y, spmv_grid.z, spmv_block.x, spmv_block.y, spmv_block.z);
        printf("[MIX] mix_grid -- %d * %d * %d mix_block -- %d * %d * %d \n", mix_grid.x, mix_grid.y, mix_grid.z, mix_block.x, mix_block.y, mix_block.z);

        cudaErrCheck(cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth*sizeof(int)));
	    cudaErrCheck(cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt,nzcnt_len*sizeof(int)));

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel0 <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// sgemm parameters
			spmv_ptb_result_vector, spmv_ptb_matrix, 
            spmv_ptb_matrix_indice, spmv_ptb_matrix_perm, 
            spmv_ptb_vector, spmv_ptb_matrix_nzcnt, dim,
            spmv_grid_dim_x, spmv_grid_dim_y, spmv_block_dim_x, spmv_block_dim_y, spmv_iter
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
		checkKernelErrors((ptb_spmv<<<spmv_grid, spmv_block, 0, streams[1]>>> (spmv_ptb_result_vector, spmv_ptb_matrix, 
                    spmv_ptb_matrix_indice, spmv_ptb_matrix_perm, 
                    spmv_ptb_vector, spmv_ptb_matrix_nzcnt, dim,
                    spmv_grid_dim_x, spmv_grid_dim_y, spmv_block_dim_x, spmv_block_dim_y, spmv_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] spmv_grid -- %d * %d * %d spmv_block -- %d * %d * %d \n", 
            spmv_grid.x, spmv_grid.y, spmv_grid.z, spmv_block.x, spmv_block.y, spmv_block.z);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 3) {
        spmv_grid.x = 1147;
        spmv_grid.y = spmv_grid.z = 1;
        spmv_block.x = 128;
        spmv_block.y = spmv_block.z = 1;

        cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_spmv<<<spmv_grid, spmv_block, 0, streams[1]>>> (spmv_ptb_result_vector, spmv_ptb_matrix, 
                    spmv_ptb_matrix_indice, spmv_ptb_matrix_perm, 
                    spmv_ptb_vector, spmv_ptb_matrix_nzcnt, dim,
                    spmv_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[STREAMO] mix took %f ms\n\n", kernel_time);
    }

    printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

	// Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_wmma_ori_c, wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_wmma_ptb_c, wmma_ptb_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_spmv_ori_result_vector, spmv_ori_result_vector, dim * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_spmv_ptb_result_vector, spmv_ptb_result_vector, dim * sizeof(float), cudaMemcpyDeviceToHost));

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
    for (int i = 0; i < dim; i++) {
        float v1 = host_spmv_ori_result_vector[i];
        float v2 = host_spmv_ptb_result_vector[i];
        if (fabs(v1 - v2) > 0.001f) {
        errors++;
        if (errors < 10) printf("%f %f\n", v1, v2);
        }
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