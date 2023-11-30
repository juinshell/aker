#include <stdio.h>
#include <curand.h>
#include <cublas_v2.h>

#include "file_t/spmv_t/convert_dataset.h"
#include "file_t/spmv_t/mmio.h"

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
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
using namespace nvcuda; // ??? 여기서 에러가 발생합니다. 

#include "header/spmv_header.h"
#include "file_t/spmv_t/spmv_kernel.cu"

int main(int argc, char* argv[]) {
    int spmv_blks = 1;
    int spmv_iter = 1;
    if (argc == 3) {
        spmv_blks = atoi(argv[1]);
        spmv_iter = atoi(argv[2]);
    } 

    // variables
    // ---------------------------------------------------------------------------------------
        float kernel_time;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));
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

        char mtx_bin[] = "file_t/spmv_t/spmv_mtx.bin";
        char vec_bin[] = "file_t/spmv_t/spmv_vec.bin";

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
    // ---------------------------------------------------------------------------------------


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

        printf("[PTB] Running with spmv...\n");
        printf("[PTB] spmv_grid -- %d * %d * %d spmv_block -- %d * %d * %d \n", 
            spmv_grid.x, spmv_grid.y, spmv_grid.z, spmv_block.x, spmv_block.y, spmv_block.z);

        cudaErrCheck(cudaEventRecord(startKERNEL));
        checkKernelErrors((ptb_spmv<<<spmv_grid, spmv_block>>> (spmv_ptb_result_vector, spmv_ptb_matrix, 
                        spmv_ptb_matrix_indice, spmv_ptb_matrix_perm, 
                        spmv_ptb_vector, spmv_ptb_matrix_nzcnt, dim,
                        spmv_grid_dim_x, spmv_grid_dim_y, spmv_block_dim_x, spmv_block_dim_y, spmv_iter)));
        cudaErrCheck(cudaEventRecord(stopKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        printf("[PTB] spmv took %f ms\n\n", kernel_time);
    // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
        printf("Checking results...\n");
        cudaErrCheck(cudaMemcpy(host_spmv_ori_result_vector, spmv_ori_result_vector, dim * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(host_spmv_ptb_result_vector, spmv_ptb_result_vector, dim * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
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
    // ---------------------------------------------------------------------------------------

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaFree(spmv_ori_matrix));
    cudaErrCheck(cudaFree(spmv_ori_matrix_indice));
    cudaErrCheck(cudaFree(spmv_ori_matrix_perm));
    cudaErrCheck(cudaFree(spmv_ori_matrix_nzcnt));
    cudaErrCheck(cudaFree(spmv_ori_vector));
    cudaErrCheck(cudaFree(spmv_ori_result_vector));

    cudaErrCheck(cudaFree(spmv_ptb_matrix));
    cudaErrCheck(cudaFree(spmv_ptb_matrix_indice));
    cudaErrCheck(cudaFree(spmv_ptb_matrix_perm));
    cudaErrCheck(cudaFree(spmv_ptb_matrix_nzcnt));
    cudaErrCheck(cudaFree(spmv_ptb_vector));
    cudaErrCheck(cudaFree(spmv_ptb_result_vector));

    free(host_spmv_ori_result_vector);
    free(host_spmv_ptb_result_vector);
    free(host_ori_vector);

    cudaErrCheck(cudaDeviceReset());
    return 0;
}