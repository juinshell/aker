#include <stdio.h>
#include "switch.h"
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
using namespace nvcuda;

#include "header/lbm_header.h"

#include "file_t/lbm_kernel.cu"

#include "cuda_ipc.h"
// Define some error checking macros.
#define cudaErrCheck(stat)                       \
   {                                             \
      cudaErrCheck_((stat), __FILE__, __LINE__); \
   }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
   if (stat != cudaSuccess)
   {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat)                       \
   {                                               \
      cublasErrCheck_((stat), __FILE__, __LINE__); \
   }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
{
   if (stat != CUBLAS_STATUS_SUCCESS)
   {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define curandErrCheck(stat)                       \
   {                                               \
      curandErrCheck_((stat), __FILE__, __LINE__); \
   }
void curandErrCheck_(curandStatus_t stat, const char *file, int line)
{
   if (stat != CURAND_STATUS_SUCCESS)
   {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}

#define checkKernelErrors(expr)                                \
   do                                                          \
   {                                                           \
      expr;                                                    \
                                                               \
      cudaError_t __err = cudaGetLastError();                  \
      if (__err != cudaSuccess)                                \
      {                                                        \
         printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
                cudaGetErrorString(__err));                    \
         abort();                                              \
      }                                                        \
   } while (0)

int main(int argc, char *argv[])
{
   int lbm_blks = 2;
   int lbm_iter = 60000;
   if (argc == 3)
   {
      lbm_blks = atoi(argv[1]);
      lbm_iter = atoi(argv[2]);
   }

   // variables
   // ---------------------------------------------------------------------------------------
   float kernel_time;
   cudaEvent_t start_kernel;
   cudaEvent_t stop_kernel;
   cudaErrCheck(cudaEventCreate(&start_kernel));
   cudaErrCheck(cudaEventCreate(&stop_kernel));

   // lbm variables
   // ---------------------------------------------------------------------------------------
   float *lbm_ori_src;
   float *lbm_ori_dst;
   float *lbm_pers_src;
   float *lbm_pers_dst;
   float *host_lbm_ori_dst;
   float *host_lbm_pers_dst;

   const size_t size = TOTAL_PADDED_CELLS * N_CELL_ENTRIES * sizeof(float) + 2 * TOTAL_MARGIN * sizeof(float);

   host_lbm_ori_dst = (float *)malloc(size);
   host_lbm_pers_dst = (float *)malloc(size);
   cudaErrCheck(cudaMalloc((void **)&lbm_ori_src, size));
   cudaErrCheck(cudaMalloc((void **)&lbm_ori_dst, size));
   cudaErrCheck(cudaMalloc((void **)&lbm_pers_src, size));
   cudaErrCheck(cudaMalloc((void **)&lbm_pers_dst, size));

   curandGenerator_t gen;
   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
   curandErrCheck(curandGenerateUniform(gen, lbm_ori_src, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
   curandErrCheck(curandGenerateUniform(gen, lbm_ori_dst, TOTAL_PADDED_CELLS * N_CELL_ENTRIES + 2 * TOTAL_MARGIN));
   cudaErrCheck(cudaMemcpy(lbm_pers_src, lbm_ori_src, size, cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(lbm_pers_dst, lbm_ori_dst, size, cudaMemcpyDeviceToDevice));
   lbm_ori_src += REAL_MARGIN;
   lbm_ori_dst += REAL_MARGIN;
   lbm_pers_src += REAL_MARGIN;
   lbm_pers_dst += REAL_MARGIN;

   // SOLO running
   // ---------------------------------------------------------------------------------------
   dim3 lbm_block, lbm_grid;
   lbm_block.x = SIZE_X;
   lbm_grid.x = SIZE_Y;
   lbm_grid.y = SIZE_Z;
   lbm_block.y = lbm_block.z = lbm_grid.z = 1;
   printf("[ORI] Running with lbm...\n");
   printf("[ORI] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);

   // int num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_lbm, lbm_block.x * lbm_block.y * lbm_block.z, 0);
   // printf("[lbm] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);

   /* Barrier */
   int fd = shm_open(MMAP_FILE, O_RDWR, 0666);
   // ftruncate(fd, sizeof(pthread_barrier_t));
   pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors((ori_lbm<<<lbm_grid, lbm_block>>>(lbm_ori_src, lbm_ori_dst, lbm_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[ORI] lbm took %f ms\n\n", kernel_time);

   // PTB running
   // ---------------------------------------------------------------------------------------
   int lbm_block_dim_x = lbm_block.x;
   int lbm_block_dim_y = lbm_block.y;
   int lbm_block_dim_z = lbm_block.z;
   int lbm_grid_dim_x = lbm_grid.x;
   int lbm_grid_dim_y = lbm_grid.y;
   int lbm_grid_dim_z = lbm_grid.z;
   // lbm_grid.x = lbm_grid_dim_x * lbm_grid_dim_y * lbm_grid_dim_z;
   // lbm_grid.x = 80 * 1;
   lbm_grid.x = lbm_blks == 0 ? lbm_grid_dim_x * lbm_grid_dim_y : 68 * lbm_blks;
   lbm_grid.y = lbm_grid.z = 1;
   lbm_block.x = lbm_block_dim_x * lbm_block_dim_y * lbm_block_dim_z;
   lbm_block.y = lbm_block.z = 1;
   printf("[PERS] Running with lbm...\n");
   printf("[PERS] lbm_grid -- %d * %d * %d lbm_block -- %d * %d * %d \n", lbm_grid.x, lbm_grid.y, lbm_grid.z, lbm_block.x, lbm_block.y, lbm_block.z);

   // num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_lbm, lbm_block.x * lbm_block.y * lbm_block.z, 0);
   // printf("[PERS] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);

   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors((pers_lbm<<<lbm_grid, lbm_block>>>(lbm_pers_src, lbm_pers_dst,
                                                        lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
                                                        lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z, lbm_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[PERS] lbm took %f ms\n\n", kernel_time);

   // cudaErrCheck(cudaEventRecord(start_kernel));
   // checkKernelErrors((pers_lbm<<<lbm_grid, lbm_block>>>(lbm_pers_src, lbm_pers_dst,
   //                                                      lbm_grid_dim_x, lbm_grid_dim_y, lbm_grid_dim_z,
   //                                                      lbm_block_dim_x, lbm_block_dim_y, lbm_block_dim_z, lbm_iter)));
   // cudaErrCheck(cudaEventRecord(stop_kernel));
   // cudaErrCheck(cudaEventSynchronize(stop_kernel));
   // cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   // printf("[PERS] lbm took %f ms\n\n", kernel_time);

   cudaErrCheck(cudaEventDestroy(start_kernel));
   cudaErrCheck(cudaEventDestroy(stop_kernel));

   // cudaErrCheck(cudaFree(lbm_ori_src));
   // cudaErrCheck(cudaFree(lbm_ori_dst));
   // cudaErrCheck(cudaFree(lbm_pers_src));
   // cudaErrCheck(cudaFree(lbm_pers_dst));

   free(host_lbm_ori_dst);
   free(host_lbm_pers_dst);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}