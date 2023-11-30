#include "switch.h"
#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>
#include "cuda_ipc.h"

#include <mma.h>
using namespace nvcuda;

#include "header/sgemm_header.h"

#include "file_t/sgemm_kernel.cu"
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

int main(int argc, char **argv)
{

   /* Prepare for sgemm */
   int sgemm_blks = 2;
   int sgemm_iter = 66;

   // variables
   float kernel_time;
   cudaEvent_t start_kernel;
   cudaEvent_t stop_kernel;
   cudaErrCheck(cudaEventCreate(&start_kernel));
   cudaErrCheck(cudaEventCreate(&stop_kernel));

   // sgemm variables
   float *sgemm_ori_a;
   float *sgemm_ori_b;
   float *sgemm_ori_c;
   float *sgemm_pers_a;
   float *sgemm_pers_b;
   float *sgemm_pers_c;
   float *host_sgemm_ori_c;
   float *host_sgemm_pers_c;

   int NORMAL_M = 4096;
   int NORMAL_N = 4128;
   int NORMAL_K = 4064;

   cudaErrCheck(cudaMalloc((void **)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&sgemm_pers_a, NORMAL_M * NORMAL_K * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&sgemm_pers_b, NORMAL_K * NORMAL_N * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&sgemm_pers_c, NORMAL_M * NORMAL_N * sizeof(float)));

   host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
   host_sgemm_pers_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
   curandGenerator_t gen;
   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
   curandErrCheck(curandGenerateUniform(gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
   curandErrCheck(curandGenerateUniform(gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
   cudaErrCheck(cudaMemcpy(sgemm_pers_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(sgemm_pers_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
   curandErrCheck(curandDestroyGenerator(gen));

   // SOLO running
   dim3 sgemm_grid;
   dim3 sgemm_block;
   sgemm_block.x = TILE_N;
   sgemm_block.y = TILE_TB_HEIGHT;
   sgemm_grid.x = NORMAL_M / TILE_M;
   sgemm_grid.y = NORMAL_N / TILE_N;
   // printf("[ORI] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);
   // int num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_sgemm, sgemm_block.x * sgemm_block.y, 0);
   // printf("[cutcp] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);

   // Barrier
   /* Get the barrier in shared memory */
   int fd = shm_open(MMAP_FILE, O_RDWR, 0666);

   // ftruncate(fd, sizeof(pthread_barrier_t));

   pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

   printf("[ORI] Running with sgemm...\n");
   pthread_barrier_wait(shared_barrier);

   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors((ori_sgemm<<<sgemm_grid, sgemm_block>>>(sgemm_ori_a, sgemm_ori_b, sgemm_ori_c,
                                                             NORMAL_M, NORMAL_N, NORMAL_K, sgemm_iter)));

   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[ORI] sgemm took %f ms\n\n", kernel_time);

   // PTB running
   // ---------------------------------------------------------------------------------------
   int sgemm_grid_dim_x = sgemm_grid.x;
   int sgemm_grid_dim_y = sgemm_grid.y;
   int sgemm_block_dim_x = sgemm_block.x;
   int sgemm_block_dim_y = sgemm_block.y;
   sgemm_grid.x = sgemm_grid_dim_x * sgemm_grid_dim_y;
   sgemm_grid.x = sgemm_blks == 0 ? sgemm_grid_dim_x * sgemm_grid_dim_y : 68 * sgemm_blks;
   sgemm_grid.y = 1;
   sgemm_block.x = sgemm_block_dim_x * sgemm_block_dim_y;
   sgemm_block.y = 1;
   printf("[PERS] Running with sgemm...\n");
   printf("[PERS] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors((pers_sgemm<<<sgemm_grid, sgemm_block>>>(sgemm_pers_a, sgemm_pers_b, sgemm_pers_c, NORMAL_M, NORMAL_N, NORMAL_K,
                                                              sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[PERS] sgemm took %f ms\n\n", kernel_time);

   // cudaErrCheck(cudaEventRecord(start_kernel));
   // checkKernelErrors((pers_sgemm<<<sgemm_grid, sgemm_block>>>(sgemm_pers_a, sgemm_pers_b, sgemm_pers_c, NORMAL_M, NORMAL_N, NORMAL_K,
   //                                                            sgemm_grid_dim_x, sgemm_grid_dim_y, sgemm_block_dim_x, sgemm_block_dim_y, sgemm_iter)));
   // cudaErrCheck(cudaEventRecord(stop_kernel));
   // cudaErrCheck(cudaEventSynchronize(stop_kernel));
   // cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   // printf("[PERS] sgemm took %f ms\n\n", kernel_time);

   cudaErrCheck(cudaEventDestroy(start_kernel));
   cudaErrCheck(cudaEventDestroy(stop_kernel));

   cudaErrCheck(cudaFree(sgemm_ori_a));
   cudaErrCheck(cudaFree(sgemm_ori_b));
   cudaErrCheck(cudaFree(sgemm_ori_c));

   return 0;
}