#include <stdio.h>
#include "switch.h"
#include <curand.h>
#include <cublas_v2.h>

#include <mma.h>
using namespace nvcuda;

#include "header/cutcp_header.h"

#include "file_t/cutcp_kernel.cu"

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
   int cutcp_blks = 2;
   int cutcp_iter = 63;
   if (argc == 3)
   {
      cutcp_blks = atoi(argv[1]);
      cutcp_iter = atoi(argv[2]);
   }

   // variables
   // ---------------------------------------------------------------------------------------
   float kernel_time;
   cudaEvent_t start_kernel;
   cudaEvent_t stop_kernel;
   cudaErrCheck(cudaEventCreate(&start_kernel));
   cudaErrCheck(cudaEventCreate(&stop_kernel));

   // cutcp variables
   // ---------------------------------------------------------------------------------------
   int nbrlistlen = 335;
   // int3 *nbrlist;
   int3 nbrlist[NBRLIST_MAXLEN];

   float *cutcp_ori_regionZeroCuda;
   float4 *cutcp_ori_binBaseCuda, *cutcp_ori_binZeroCuda, *host_cutcp_ori_binBaseCuda;
   float *cutcp_pers_regionZeroCuda;
   float4 *cutcp_pers_binBaseCuda, *cutcp_pers_binZeroCuda, *host_cutcp_pers_binBaseCuda;

   int lnx = 208;
   int lny = 208;
   int lnz = 208;
   int lnall = lnx * lny * lnz;
   int nbins = 32768;

   int zRegionDim = 26;
   int binDim_x = 32;
   int binDim_y = 32;
   float h = 0.5;
   float cutoff2 = 144.0;
   float inv_cutoff2 = 0.006944;

   // nbrlist = (int3 *)malloc(NBRLIST_MAXLEN * sizeof(int3));
   // memset(nbrlist, 1, NBRLIST_MAXLEN * sizeof(int3));
   for (int i = 0; i < NBRLIST_MAXLEN; i++)
   {
      nbrlist[i].x = i % 7 - 3;
      nbrlist[i].y = i % 7 - 3;
      nbrlist[i].z = i % 7 - 3;
   }

   cudaErrCheck(cudaMalloc((void **)&cutcp_ori_regionZeroCuda, lnall * sizeof(float)));
   cudaErrCheck(cudaMalloc((void **)&cutcp_pers_regionZeroCuda, lnall * sizeof(float)));
   cudaErrCheck(cudaMemset(cutcp_ori_regionZeroCuda, 2, lnall * sizeof(float)));
   cudaErrCheck(cudaMemset(cutcp_pers_regionZeroCuda, 2, lnall * sizeof(float)));

   cudaErrCheck(cudaMalloc((void **)&cutcp_ori_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
   cudaErrCheck(cudaMalloc((void **)&cutcp_pers_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
   cudaErrCheck(cudaMemset(cutcp_ori_binBaseCuda, 3, nbins * BIN_DEPTH * sizeof(float4)));
   cudaErrCheck(cudaMemset(cutcp_pers_binBaseCuda, 3, nbins * BIN_DEPTH * sizeof(float4)));
   cutcp_ori_binZeroCuda = cutcp_ori_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;
   cutcp_pers_binZeroCuda = cutcp_pers_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;

   host_cutcp_ori_binBaseCuda = (float4 *)malloc(nbins * BIN_DEPTH * sizeof(float4));
   host_cutcp_pers_binBaseCuda = (float4 *)malloc(nbins * BIN_DEPTH * sizeof(float4));

   cudaErrCheck(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
   cudaErrCheck(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));

   // SOLO running
   // ---------------------------------------------------------------------------------------
   dim3 cutcp_grid, cutcp_block;
   cutcp_grid.x = zRegionDim;
   cutcp_grid.y = zRegionDim;
   cutcp_grid.z = zRegionDim;
   cutcp_block.x = 8;
   cutcp_block.y = 2;
   cutcp_block.z = 8;

   printf("[ORI] Running with cutcp...\n");
   printf("[ORI] cutcp_grid -- %d * %d * %d cutcp_block -- %d * %d * %d \n", cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, cutcp_block.x, cutcp_block.y, cutcp_block.z);
   // int num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, ori_cutcp, cutcp_block.x * cutcp_block.y * cutcp_block.z, 0);
   // printf("[cutcp] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);

   /* Barrier */
   int fd = shm_open(MMAP_FILE, O_CREAT | O_RDWR, 0666);
   // ftruncate(fd, sizeof(pthread_barrier_t));

   pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

   // pthread_barrierattr_t barrier_attr;
   // pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED);
   // pthread_barrier_init(shared_barrier, &barrier_attr, 2);

   pthread_barrier_wait(shared_barrier);

   cudaErrCheck(cudaEventRecord(start_kernel));
   checkKernelErrors((ori_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_ori_binZeroCuda, h, cutoff2, inv_cutoff2, cutcp_ori_regionZeroCuda, 25, cutcp_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[ORI] cutcp took %f ms\n\n", kernel_time);

   // PTB
   // ---------------------------------------------------------------------------------
   int cutcp_grid_dim_x = cutcp_grid.x;
   int cutcp_grid_dim_y = cutcp_grid.y;
   int cutcp_grid_dim_z = cutcp_grid.z;
   int cutcp_block_dim_x = cutcp_block.x;
   int cutcp_block_dim_y = cutcp_block.y;
   int cutcp_block_dim_z = cutcp_block.z;
   // cutcp_grid.x = cutcp_grid_dim_x * cutcp_grid_dim_y * cutcp_grid_dim_z;
   cutcp_grid.x = cutcp_blks == 0 ? cutcp_grid_dim_x * cutcp_grid_dim_y * cutcp_grid_dim_z : 68 * cutcp_blks;
   cutcp_grid.y = 1;
   cutcp_grid.z = 1;
   cutcp_block.x = cutcp_block_dim_x * cutcp_block_dim_y * cutcp_block_dim_z;
   cutcp_block.y = 1;
   cutcp_block.z = 1;

   cudaErrCheck(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
   cudaErrCheck(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));

   printf("[PERS] Running with cutcp...\n");
   printf("[PERS] cutcp_grid -- %d * %d * %d cutcp_block -- %d * %d * %d \n", cutcp_grid.x, cutcp_grid.y, cutcp_grid.z, cutcp_block.x, cutcp_block.y, cutcp_block.z);
   // num_blocks = 0;
   // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, pers_cutcp, cutcp_block.x * cutcp_block.y * cutcp_block.z, 0);
   // printf("[PERS] cudaOccupancyMaxActiveBlocksPerMultiprocessor: %d\n", num_blocks);
   pthread_barrier_wait(shared_barrier);
   cudaErrCheck(cudaEventRecord(start_kernel));
   // checkKernelErrors((pers_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_pers_binZeroCuda,
   //     h, cutoff2, inv_cutoff2, cutcp_pers_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
   //     cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter)));

   checkKernelErrors((pers_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_pers_binZeroCuda,
                                                              h, cutoff2, inv_cutoff2, cutcp_pers_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
                                                              cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter)));
   cudaErrCheck(cudaEventRecord(stop_kernel));
   cudaErrCheck(cudaEventSynchronize(stop_kernel));
   cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   printf("[PERS] cutcp took %f ms\n\n", kernel_time);


   // cudaErrCheck(cudaEventRecord(start_kernel));
   // // checkKernelErrors((pers_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_pers_binZeroCuda,
   // //     h, cutoff2, inv_cutoff2, cutcp_pers_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
   // //     cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter)));

   // checkKernelErrors((pers_cutcp<<<cutcp_grid, cutcp_block>>>(binDim_x, binDim_y, cutcp_pers_binZeroCuda,
   //                                                            h, cutoff2, inv_cutoff2, cutcp_pers_regionZeroCuda, 25, cutcp_grid_dim_x, cutcp_grid_dim_y, cutcp_grid_dim_z,
   //                                                            cutcp_block_dim_x, cutcp_block_dim_y, cutcp_block_dim_z, cutcp_iter)));
   // cudaErrCheck(cudaEventRecord(stop_kernel));
   // cudaErrCheck(cudaEventSynchronize(stop_kernel));
   // cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
   // printf("[PERS] cutcp took %f ms\n\n", kernel_time);

   cudaErrCheck(cudaEventDestroy(start_kernel));
   cudaErrCheck(cudaEventDestroy(stop_kernel));

   cudaErrCheck(cudaFree(cutcp_ori_regionZeroCuda));
   cudaErrCheck(cudaFree(cutcp_ori_binBaseCuda));
   cudaErrCheck(cudaFree(cutcp_pers_regionZeroCuda));
   cudaErrCheck(cudaFree(cutcp_pers_binBaseCuda));

   free(host_cutcp_ori_binBaseCuda);
   free(host_cutcp_pers_binBaseCuda);

   cudaErrCheck(cudaDeviceReset());
   return 0;
}