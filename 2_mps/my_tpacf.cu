#include <stdio.h>
#include "switch.h"
#include <curand.h>
#include <cublas_v2.h>

#include <mma.h>
using namespace nvcuda;

#include "header/tpacf_header.h"

#include "file_t/tpacf_kernel.cu"

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
   int tpacf_blks = 3;
   int tpacf_iter = 90;
   if (argc == 3)
   {
      tpacf_blks = atoi(argv[1]);
      tpacf_iter = atoi(argv[2]);
   }

   // variables
   // ---------------------------------------------------------------------------------------
   float kernel_time;
   curandGenerator_t gen;
   cudaEvent_t start_kernel;
   cudaEvent_t stop_kernel;
   cudaErrCheck(cudaEventCreate(&start_kernel));
   cudaErrCheck(cudaEventCreate(&stop_kernel));

   // tpacf variables
   // ---------------------------------------------------------------------------------------
   // 10391
   // 97178
   // NUM_ELEMENTS = 97178;
   int NUM_ELEMENTS = 9718;
   NUM_ELEMENTS = 2048;
   int NUM_SETS = 100;
   int num_elements = NUM_ELEMENTS;
   unsigned f_mem_size = (1 + NUM_SETS) * num_elements * sizeof(float);
   float *binb = (float *)malloc((NUM_BINS + 1) * sizeof(float));
   for (int k = 0; k < NUM_BINS + 1; k++)
   {
      binb[k] = cos(pow(10.0, (log10(min_arcmin) + k * 1.0 / bins_per_dec)) / 60.0 * D2R);
   }

   hist_t *tpacf_ori_hists;
   float *tpacf_ori_x;
   float *tpacf_ori_y;
   float *tpacf_ori_z;
   hist_t *tpacf_pers_hists;
   float *tpacf_pers_x;
   float *tpacf_pers_y;
   float *tpacf_pers_z;
   hist_t *host_tpacf_ori_hists;
   hist_t *host_tpacf_pers_hists;

   {
      cudaErrCheck(cudaMalloc((void **)&tpacf_ori_hists, NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t)));
      cudaErrCheck(cudaMemset(tpacf_ori_hists, 100, NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t)));
      cudaErrCheck(cudaMalloc((void **)&tpacf_ori_x, f_mem_size));
      cudaErrCheck(cudaMalloc((void **)&tpacf_ori_y, f_mem_size));
      cudaErrCheck(cudaMalloc((void **)&tpacf_ori_z, f_mem_size));

      cudaErrCheck(cudaMalloc((void **)&tpacf_pers_hists, NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t)));
      cudaErrCheck(cudaMemset(tpacf_pers_hists, 100, NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t)));
      cudaErrCheck(cudaMalloc((void **)&tpacf_pers_x, f_mem_size));
      cudaErrCheck(cudaMalloc((void **)&tpacf_pers_y, f_mem_size));
      cudaErrCheck(cudaMalloc((void **)&tpacf_pers_z, f_mem_size));

      host_tpacf_ori_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t));
      host_tpacf_pers_hists = (hist_t *)malloc(NUM_BINS * (NUM_SETS * 2 + 1) * sizeof(hist_t));

      curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
      curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
      curandErrCheck(curandGenerateUniform(gen, tpacf_ori_x, (1 + NUM_SETS) * num_elements));
      curandErrCheck(curandGenerateUniform(gen, tpacf_ori_y, (1 + NUM_SETS) * num_elements));
      curandErrCheck(curandGenerateUniform(gen, tpacf_ori_z, (1 + NUM_SETS) * num_elements));

      cudaErrCheck(cudaMemcpy(tpacf_pers_x, tpacf_ori_x, f_mem_size, cudaMemcpyDeviceToDevice));
      cudaErrCheck(cudaMemcpy(tpacf_pers_y, tpacf_ori_y, f_mem_size, cudaMemcpyDeviceToDevice));
      cudaErrCheck(cudaMemcpy(tpacf_pers_z, tpacf_ori_z, f_mem_size, cudaMemcpyDeviceToDevice));
   }

   // SOLO running
   // ---------------------------------------------------------------------------------------
   dim3 tpacf_grid;
   dim3 tpacf_block;
   tpacf_block.x = BLOCK_SIZE;
   tpacf_block.y = 1;
   tpacf_grid.x = NUM_SETS * 2 + 1;
   tpacf_grid.y = 1;
   printf("[ORI] Running with tpacf...\n");
   printf("[ORI] tpacf_grid -- %d * %d * %d tpacf_block -- %d * %d * %d \n",
          tpacf_grid.x, tpacf_grid.y, tpacf_grid.z, tpacf_block.x, tpacf_block.y, tpacf_block.z);

   // int num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_tpacf, tpacf_block.x, 0);
   // printf("[ORI] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);
   cudaMemcpyToSymbol(dev_binb, binb, (NUM_BINS + 1) * sizeof(float));

   /* Barrier */
   int fd = shm_open(MMAP_FILE, O_RDWR, 0666);
   // ftruncate(fd, sizeof(pthread_barrier_t));

   pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors((ori_tpacf<<<tpacf_grid, tpacf_block>>>(tpacf_ori_hists, tpacf_ori_x, tpacf_ori_y, tpacf_ori_z,
                                                             NUM_SETS, NUM_ELEMENTS, tpacf_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[ORI] tpacf took %f ms\n\n", kernel_time);

   // PTB
   // ---------------------------------------------------------------------------------
   int tpacf_grid_dim_x = tpacf_grid.x;
   int tpacf_grid_dim_y = tpacf_grid.y;
   int tpacf_block_dim_x = tpacf_block.x;
   int tpacf_block_dim_y = tpacf_block.y;
   tpacf_grid.x = tpacf_blks == 0 ? tpacf_grid_dim_x * tpacf_grid_dim_y : 68 * tpacf_blks;
   tpacf_grid.y = 1;
   tpacf_block.x = tpacf_block_dim_x * tpacf_block_dim_y;
   tpacf_block.y = 1;
   printf("[PERS] Running with tpacf...\n");
   printf("[PERS] tpacf_grid -- %d * %d * %d tpacf_block -- %d * %d * %d \n", tpacf_grid.x, tpacf_grid.y, tpacf_grid.z, tpacf_block.x, tpacf_block.y, tpacf_block.z);

   // num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, pers_tpacf, tpacf_block.x, 0);
   // printf("[PERS] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);

   cudaMemcpyToSymbol(dev_binb, binb, (NUM_BINS + 1) * sizeof(float));

   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors((pers_tpacf<<<tpacf_grid, tpacf_block>>>(tpacf_pers_hists, tpacf_pers_x, tpacf_pers_y, tpacf_pers_z,
                                                              NUM_SETS, NUM_ELEMENTS, tpacf_grid_dim_x, tpacf_grid_dim_y, tpacf_block_dim_x, tpacf_block_dim_y, tpacf_iter)));
   // checkKernelErrors((ori_tpacf <<< tpacf_grid, tpacf_block >>> (tpacf_pers_hists, tpacf_pers_x, tpacf_pers_y, tpacf_pers_z,
   //                     NUM_SETS, NUM_ELEMENTS, tpacf_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[PERS] tpacf took %f ms\n\n", kernel_time);

   // cudaErrCheck(cudaEventRecord(start_kernel));
   // checkKernelErrors((pers_tpacf<<<tpacf_grid, tpacf_block>>>(tpacf_pers_hists, tpacf_pers_x, tpacf_pers_y, tpacf_pers_z,
   //                                                            NUM_SETS, NUM_ELEMENTS, tpacf_grid_dim_x, tpacf_grid_dim_y, tpacf_block_dim_x, tpacf_block_dim_y, tpacf_iter)));
   // // checkKernelErrors((ori_tpacf <<< tpacf_grid, tpacf_block >>> (tpacf_pers_hists, tpacf_pers_x, tpacf_pers_y, tpacf_pers_z,
   // //                     NUM_SETS, NUM_ELEMENTS, tpacf_iter)));
   // cudaErrCheck(cudaEventRecord(stop_kernel));
   // cudaErrCheck(cudaEventSynchronize(stop_kernel));
   // cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   // printf("[PERS] tpacf took %f ms\n\n", kernel_time);

   cudaErrCheck(cudaEventDestroy(start_kernel));
   cudaErrCheck(cudaEventDestroy(stop_kernel));

   cudaErrCheck(cudaDeviceReset());
   return 0;
}