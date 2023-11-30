#include <stdio.h>
#include "switch.h"
#include <curand.h>
#include <cublas_v2.h>

#include <mma.h>
using namespace nvcuda;

#include "header/regtiling_header.h"

#include "file_t/regtiling_kernel.cu"

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
   int regtiling_blks = 1;
   int regtiling_iter = 6000;
   if (argc == 3)
   {
      regtiling_blks = atoi(argv[1]);
      regtiling_iter = atoi(argv[2]);
   }

   // variables
   // ---------------------------------------------------------------------------------------
   float kernel_time;
   cudaEvent_t start_kernel;
   cudaEvent_t stop_kernel;
   cudaErrCheck(cudaEventCreate(&start_kernel));
   cudaErrCheck(cudaEventCreate(&stop_kernel));

   // regtiling variables
   // ---------------------------------------------------------------------------------------
   float *host_regtiling_ori_a0;
   float *regtiling_ori_a0;
   float *regtiling_ori_anext;

   float *host_regtiling_pers_a0;
   float *regtiling_pers_a0;
   float *regtiling_pers_anext;

   float c0 = 1.0f / 6.0f;
   float c1 = 1.0f / 6.0f / 6.0f;

   // nx = 128 ny = 128 nz = 32 iter = 100
   // nx = 512 ny = 512 nz = 64 iter = 100
   int nx = 128 * 4;
   int ny = 128 * 4;
   int nz = 32 * 2;

   // printf("nx: %d, ny: %d, nz: %d, iteration: %d \n", nx, ny, nz, iteration);
   host_regtiling_ori_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
   cudaErrCheck(cudaMalloc((void **)&regtiling_ori_a0, nx * ny * nz * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&regtiling_ori_anext, nx * ny * nz * sizeof(float)));

   host_regtiling_pers_a0 = (float *)malloc(nx * ny * nz * sizeof(float));
   cudaErrCheck(cudaMalloc((void **)&regtiling_pers_a0, nx * ny * nz * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&regtiling_pers_anext, nx * ny * nz * sizeof(float)));

   curandGenerator_t gen;
   curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   curandErrCheck(curandGenerateUniform(gen, regtiling_ori_a0, nx * ny * nz));
   cudaErrCheck(cudaMemcpy(regtiling_ori_anext, regtiling_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(regtiling_pers_a0, regtiling_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));
   cudaErrCheck(cudaMemcpy(regtiling_pers_anext, regtiling_ori_a0, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToDevice));

   // SOLO running
   // ---------------------------------------------------------------------------------------
   dim3 regtiling_grid;
   dim3 regtiling_block;
   regtiling_block.x = tile_x;
   regtiling_block.y = tile_y;
   regtiling_grid.x = (nx + tile_x * 2 - 1) / (tile_x * 2);
   regtiling_grid.y = (ny + tile_y - 1) / tile_y;

   printf("[ORI] Running with regtiling...\n");
   printf("[ORI] regtiling_grid -- %d * %d * %d regtiling_block -- %d * %d * %d \n",
          regtiling_grid.x, regtiling_grid.y, regtiling_grid.z, regtiling_block.x, regtiling_block.y, regtiling_block.z);
   // int num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_regtiling, regtiling_block.x * regtiling_block.y * regtiling_block.z, 0);
   // printf("[ORI] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);

   /* Barrier */
   int fd = shm_open(MMAP_FILE, O_RDWR, 0666);
   // ftruncate(fd, sizeof(pthread_barrier_t));

   pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors(
       (ori_regtiling<<<regtiling_grid, regtiling_block>>>(c0, c1, regtiling_ori_a0, regtiling_ori_anext, nx, ny, nz, regtiling_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[ORI] regtiling took %f ms\n\n", kernel_time);

   // PTB running
   // ---------------------------------------------------------------------------------------
   int regtiling_grid_dim_x = regtiling_grid.x;
   int regtiling_grid_dim_y = regtiling_grid.y;
   int regtiling_block_dim_x = regtiling_block.x;
   int regtiling_block_dim_y = regtiling_block.y;
   regtiling_grid.x = regtiling_blks == 0 ? regtiling_grid_dim_x * regtiling_grid_dim_y : 68 * regtiling_blks;
   regtiling_grid.y = 1;
   regtiling_block.x = regtiling_block_dim_x * regtiling_block_dim_y;
   regtiling_block.y = 1;

   printf("[PERS] Running with regtiling...\n");
   printf("[PERS] regtiling_grid -- %d * %d * %d regtiling_block -- %d * %d * %d \n",
          regtiling_grid.x, regtiling_grid.y, regtiling_grid.z, regtiling_block.x, regtiling_block.y, regtiling_block.z);
   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors(
       (pers_regtiling<<<regtiling_grid, regtiling_block>>>(c0, c1, regtiling_pers_a0, regtiling_pers_anext, nx, ny, nz,
                                                            regtiling_grid_dim_x, regtiling_grid_dim_y, regtiling_block_dim_x, regtiling_block_dim_y, regtiling_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[PERS] regtiling took %f ms\n\n", kernel_time);

   // cudaErrCheck(cudaEventRecord(start_kernel));
   // checkKernelErrors(
   //     (pers_regtiling<<<regtiling_grid, regtiling_block>>>(c0, c1, regtiling_pers_a0, regtiling_pers_anext, nx, ny, nz,
   //                                                          regtiling_grid_dim_x, regtiling_grid_dim_y, regtiling_block_dim_x, regtiling_block_dim_y, regtiling_iter)));
   // cudaErrCheck(cudaEventRecord(stop_kernel));
   // cudaErrCheck(cudaEventSynchronize(stop_kernel));
   // cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   // printf("[PERS] regtiling took %f ms\n\n", kernel_time);

   cudaErrCheck(cudaEventDestroy(start_kernel));
   cudaErrCheck(cudaEventDestroy(stop_kernel));

   cudaErrCheck(cudaFree(regtiling_ori_a0));
   cudaErrCheck(cudaFree(regtiling_ori_anext));
   cudaErrCheck(cudaFree(regtiling_pers_a0));
   cudaErrCheck(cudaFree(regtiling_pers_anext));

   free(host_regtiling_ori_a0);
   free(host_regtiling_pers_a0);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}