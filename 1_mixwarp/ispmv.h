#include <stdio.h>
#include <assert.h>
#include <curand.h>
#include <cublas_v2.h>


// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
inline void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
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

#include "header/spmv_header.h"
#include "file_t/spmv_t/convert_dataset.h"
#include "file_t/spmv_t/mmio.h"
#include "file_t/spmv_t/spmv_kernel.cu"


float ispmv_call(cudaStream_t stream, int type_id = 1) { // type_id: 0 - ori, 1 - ptb
    static int spmv_blks = 2;
    static int spmv_iter = 1700;
	static int wmma_blks = 1;
    static int wmma_iter = 100;
    static int M_INPUT = 128 * 1;
	static int N_INPUT = 128 * 3136;
	static int K_INPUT = 128 * 1;
	static int mixwarp = 2;

    // variables
    // ---------------------------------------------------------------------------------------
    static float kernel_time;
    static float serial_time = 0;
    static cudaEvent_t startKERNEL;
    static cudaEvent_t stopKERNEL;

    static bool init = false;
    if (!init) {
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
        cudaStream_t streams[2];
        for (int i = 0; i < 2; i++) {
            cudaErrCheck(cudaStreamCreate(&streams[i]));
        }
    }

    // spmv variables
    // ---------------------------------------------------------------------------------------

    //parameters declaration
    static int len;
    static int depth;
    static int dim;
    static int pad=32;
    static int nzcnt_len;
    
    //host memory allocation
    //matrix
    static float *h_data;
    static int *h_indices;
    static int *h_ptr;
    static int *h_perm;
    static int *h_nzcnt;
    //vector
    static float *host_spmv_ori_result_vector;
    static float *host_spmv_ptb_result_vector;
    static int *host_spmv_ptb_result_vector_int;
    static float *host_ori_vector;

    static char mtx_bin[] = "../0_mybench/file_t/spmv_t/spmv_mtx.bin";
    static char vec_bin[] = "../0_mybench/file_t/spmv_t/spmv_vec.bin";

        //device memory allocation
        //matrix
        static float *spmv_ori_matrix;
        static int *spmv_ori_matrix_indice;
        static int *spmv_ori_matrix_perm;
        static int *spmv_ori_matrix_nzcnt;
        //vector
        static float *spmv_ori_result_vector;
        static float *spmv_ori_vector;
        //matrix
        static float *spmv_ptb_matrix;
        static int *spmv_ptb_matrix_int;
        static int *spmv_ptb_matrix_indice;
        static int *spmv_ptb_matrix_perm;
        static int *spmv_ptb_matrix_nzcnt;
        //vector
        static float *spmv_ptb_result_vector;
        static float *spmv_ptb_vector;
        static int *spmv_ptb_result_vector_int;
        static int *spmv_ptb_vector_int;

        static int col_count;
        if (!init) {

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
            host_spmv_ptb_result_vector_int = (int*)malloc(sizeof(int)*dim);
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

        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_matrix_int, len*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_result_vector_int,dim*sizeof(int)));
        cudaErrCheck(cudaMalloc((void **)&spmv_ptb_vector_int, dim*sizeof(int)));
        cudaErrCheck(cudaMemset((void *)spmv_ptb_result_vector_int, 0, dim*sizeof(float)));


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

        cudaErrCheck(cudaMemcpy(spmv_ptb_matrix_int, h_data, len*sizeof(float), cudaMemcpyHostToDevice));
        }
    // ---------------------------------------------------------------------------------------

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
    // printf("_thread %d _block %d \n", _thread, _block);

    static dim3 spmv_grid;
    static dim3 spmv_block;
    spmv_grid.x = 1147;
    spmv_grid.y = spmv_grid.z = 1;
    spmv_block.x = 128;
    spmv_block.y = spmv_block.z = 1;

    // spmv_grid.x = _block;
    // spmv_grid.y = spmv_grid.z = 1;
    // spmv_block.x = _thread;
    // spmv_block.y = spmv_block.z = 1;

    // cudaErrCheck(cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth*sizeof(int)));
	// cudaErrCheck(cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt,nzcnt_len*sizeof(int)));

    if (type_id == 0) {
        printf("[ORI] Running with spmv...\n");
        printf("[ORI] spmv_grid -- %d * %d * %d spmv_block -- %d * %d * %d \n", 
            spmv_grid.x, spmv_grid.y, spmv_grid.z, spmv_block.x, spmv_block.y, spmv_block.z);
        
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ori_spmv<<<spmv_grid, spmv_block, 0, stream>>> (spmv_ori_result_vector, spmv_ori_matrix, 
            spmv_ori_matrix_indice, spmv_ori_matrix_perm, 
            spmv_ori_vector, spmv_ori_matrix_nzcnt, dim, spmv_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[ORI] spmv took %f ms\n\n", kernel_time);
    } else {
        int spmv_grid_dim_x = spmv_grid.x;
        int spmv_grid_dim_y = spmv_grid.y;
        int spmv_block_dim_x = spmv_block.x;
        int spmv_block_dim_y = spmv_block.y;
        spmv_grid.x = spmv_blks == 0 ? spmv_grid_dim_x * spmv_grid_dim_y : 68 * spmv_blks;
        spmv_grid.y = 1;
        spmv_block.x = spmv_block_dim_x * spmv_block_dim_y;
        spmv_block.y = 1;

        printf("[PTB] Running with spmv...\n");
        printf("[PTB] spmv_grid -- %d * %d * %d spmv_block -- %d * %d * %d \n", 
            spmv_grid.x, spmv_grid.y, spmv_grid.z, spmv_block.x, spmv_block.y, spmv_block.z);
        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_spmv<<<spmv_grid, spmv_block, 0, stream>>> (spmv_ptb_result_vector, spmv_ptb_matrix, 
                        spmv_ptb_matrix_indice, spmv_ptb_matrix_perm, 
                        spmv_ptb_vector, spmv_ptb_matrix_nzcnt, dim,
                        spmv_grid_dim_x, spmv_grid_dim_y, spmv_block_dim_x, spmv_block_dim_y, spmv_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] spmv took %f ms\n\n", kernel_time);
    }

    return kernel_time;
}