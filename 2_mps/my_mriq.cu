#include <malloc.h>
#include <stdio.h>
#include "switch.h"
#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
using namespace nvcuda;
#include "header/mriq_header.h"

#include "file_t/mriq_kernel.cu"

#include "cuda_ipc.h"

// Define some error checking macros.
#define cudaErrCheck(stat)                     \
  {                                            \
    cudaErrCheck_((stat), __FILE__, __LINE__); \
  }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
  if (stat != cudaSuccess)
  {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#define curandErrCheck(stat)                     \
  {                                              \
    curandErrCheck_((stat), __FILE__, __LINE__); \
  }
void curandErrCheck_(curandStatus_t stat, const char *file, int line)
{
  if (stat != CURAND_STATUS_SUCCESS)
  {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}

#define checkKernelErrors(expr)                             \
  do                                                        \
  {                                                         \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess)                               \
    {                                                       \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

int main(int argc, char *argv[])
{
  int mriq_blks = 2;
  int mriq_iter = 96;
  if (argc == 3)
  {
    mriq_blks = atoi(argv[1]);
    mriq_iter = atoi(argv[2]);
  }

  // variables
  // ---------------------------------------------------------------------------------------
  float kernel_time;
  cudaEvent_t start_kernel;
  cudaEvent_t stop_kernel;
  cudaErrCheck(cudaEventCreate(&start_kernel));
  cudaErrCheck(cudaEventCreate(&stop_kernel));

  // mriq variables
  // ---------------------------------------------------------------------------------------
  int numK = 2097152;
  int numX = 2097152;
  float *base_kx, *base_ky, *base_kz; /* K trajectory (3D vectors) */
  float *base_x, *base_y, *base_z;    /* X coordinates (3D vectors) */
  float *base_phiR, *base_phiI;       /* Phi values (complex) */
  // float *base_phiMag;		                /* Magnitude of Phi */
  // float *base_Qr, *base_Qi;		        /* Q signal (complex) */
  struct kValues *kVals;

  // kernel 1
  float *mriq_ori_phiR, *mriq_ori_phiI;
  float *mriq_ori_phiMag, *host_mriq_ori_phiMag;
  // kernel 2
  float *mriq_ori_x, *mriq_ori_y, *mriq_ori_z;
  float *mriq_ori_Qr, *mriq_ori_Qi, *host_mriq_ori_Qi;

  // // kernel 1
  // float *pers_phiR, *pers_phiI;
  // float *pers_phiMag, *host_pers_phiMag;
  // kernel 2
  float *mriq_pers_x, *mriq_pers_y, *mriq_pers_z;
  float *mriq_pers_Qr, *mriq_pers_Qi, *host_mriq_pers_Qi;

  inputData(&numK, &numX,
            &base_kx, &base_ky, &base_kz,
            &base_x, &base_y, &base_z,
            &base_phiR, &base_phiI);
  numK = 2097152;

  {
    // Memory allocation
    // base_phiMag = (float* ) memalign(16, numK * sizeof(float));
    // base_Qr = (float*) memalign(16, numX * sizeof (float));
    // base_Qi = (float*) memalign(16, numX * sizeof (float));
    cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiR, numK * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiI, numK * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_ori_phiMag, numK * sizeof(float)));
    host_mriq_ori_phiMag = (float *)memalign(16, numK * sizeof(float));
    cudaErrCheck(cudaMemcpy(mriq_ori_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mriq_ori_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc((void **)&mriq_ori_x, numX * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_ori_y, numX * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_ori_z, numX * sizeof(float)));
    cudaErrCheck(cudaMemcpy(mriq_ori_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mriq_ori_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mriq_ori_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc((void **)&mriq_ori_Qr, numX * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_ori_Qi, numX * sizeof(float)));
    cudaMemset((void *)mriq_ori_Qr, 0, numX * sizeof(float));
    cudaMemset((void *)mriq_ori_Qi, 0, numX * sizeof(float));
    host_mriq_ori_Qi = (float *)memalign(16, numX * sizeof(float));

    // cudaErrCheck(cudaMalloc((void **)&pers_phiR, numK * sizeof(float)));
    // cudaErrCheck(cudaMalloc((void **)&pers_phiI, numK * sizeof(float)));
    // cudaErrCheck(cudaMalloc((void **)&pers_phiMag, numK * sizeof(float)));
    // host_pers_phiMag = (float* ) memalign(16, numK * sizeof(float));
    // cudaErrCheck(cudaMemcpy(pers_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
    // cudaErrCheck(cudaMemcpy(pers_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));

    cudaErrCheck(cudaMalloc((void **)&mriq_pers_x, numX * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_pers_y, numX * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_pers_z, numX * sizeof(float)));
    cudaErrCheck(cudaMemcpy(mriq_pers_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mriq_pers_y, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(mriq_pers_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc((void **)&mriq_pers_Qr, numX * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **)&mriq_pers_Qi, numX * sizeof(float)));
    cudaMemset((void *)mriq_pers_Qr, 0, numX * sizeof(float));
    cudaMemset((void *)mriq_pers_Qi, 0, numX * sizeof(float));
    host_mriq_pers_Qi = (float *)memalign(16, numX * sizeof(float));
  }

  // SOLO running
  // ---------------------------------------------------------------------------------------
  dim3 mriq_grid1;
  dim3 mriq_block1;
  mriq_grid1.x = numK / KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  mriq_grid1.y = 1;
  mriq_block1.x = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  mriq_block1.y = 1;
  printf("[ORI] Running with mriq...\n");
  printf("[ORI] mriq_grid1 -- %d * %d * %d mriq_block1 -- %d * %d * %d \n",
         mriq_grid1.x, mriq_grid1.y, mriq_grid1.z, mriq_block1.x, mriq_block1.y, mriq_block1.z);

  checkKernelErrors((ori_ComputePhiMag<<<mriq_grid1, mriq_block1>>>(mriq_ori_phiR, mriq_ori_phiI, mriq_ori_phiMag, numK)));
  cudaMemcpy(host_mriq_ori_phiMag, mriq_ori_phiMag, numK * sizeof(float), cudaMemcpyDeviceToHost);

  kVals = (struct kValues *)calloc(numK, sizeof(struct kValues));
  for (int k = 0; k < numK; k++)
  {
    kVals[k].Kx = base_kx[k];
    kVals[k].Ky = base_ky[k];
    kVals[k].Kz = base_kz[k];
    kVals[k].PhiMag = host_mriq_ori_phiMag[k];
  }

  dim3 mriq_grid2;
  dim3 mriq_block2;
  mriq_grid2.x = numX / KERNEL_Q_THREADS_PER_BLOCK;
  mriq_grid2.y = 1;
  mriq_block2.x = KERNEL_Q_THREADS_PER_BLOCK;
  mriq_block2.y = 1;
  printf("[ORI] mriq_grid2 -- %d * %d * %d mriq_block2 -- %d * %d * %d \n",
         mriq_grid2.x, mriq_grid2.y, mriq_grid2.z, mriq_block2.x, mriq_block2.y, mriq_block2.z);

  int QGridBase = 0 * KERNEL_Q_K_ELEMS_PER_GRID;
  kValues *kValsTile = kVals + QGridBase;
  cudaMemcpyToSymbol(ck, kValsTile, KERNEL_Q_K_ELEMS_PER_GRID * sizeof(kValues), 0);

  /* Barrier */
  int fd = shm_open(MMAP_FILE, O_RDWR, 0666);
  // ftruncate(fd, MMAP_SIZE);

  pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  pthread_barrier_wait(shared_barrier);

  cudaErrCheck(cudaEventRecord(start_kernel));
  checkKernelErrors((ori_mriq<<<mriq_grid2, mriq_block2>>>(numK, QGridBase, mriq_ori_x, mriq_ori_y, mriq_ori_z, mriq_ori_Qr, mriq_ori_Qi,
                                                           mriq_iter)));
  cudaErrCheck(cudaEventRecord(stop_kernel));
  cudaErrCheck(cudaEventSynchronize(stop_kernel));
  cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
  printf("[ORI] mriq took %f ms\n\n", kernel_time);

  // PTB running
  // ---------------------------------------------------------------------------------------
  int mriq_grid2_dim_x = mriq_grid2.x;
  int mriq_block2_dim_x = mriq_block2.x;
  mriq_grid2.x = 68 * 2;
  mriq_grid2.x = mriq_blks == 0 ? mriq_grid2_dim_x : 68 * mriq_blks;
  printf("[PERS] Running with mriq...\n");
  printf("[PERS] mriq_grid2 -- %d * %d * %d mriq_block2 -- %d * %d * %d \n",
         mriq_grid2.x, mriq_grid2.y, mriq_grid2.z, mriq_block2.x, mriq_block2.y, mriq_block2.z);

  pthread_barrier_wait(shared_barrier);
  cudaErrCheck(cudaEventRecord(start_kernel));
  checkKernelErrors((pers_mriq<<<mriq_grid2, mriq_block2>>>(numK, QGridBase, mriq_pers_x, mriq_pers_y, mriq_pers_z, mriq_pers_Qr, mriq_pers_Qi,
                                                            mriq_grid2_dim_x, mriq_block2_dim_x,
                                                            mriq_iter)));
  cudaErrCheck(cudaEventRecord(stop_kernel));
  cudaErrCheck(cudaEventSynchronize(stop_kernel));
  cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
  printf("[ORI] mriq took %f ms\n\n", kernel_time);

  // cudaErrCheck(cudaEventRecord(start_kernel));
  // checkKernelErrors((pers_mriq<<<mriq_grid2, mriq_block2>>>(numK, QGridBase, mriq_pers_x, mriq_pers_y, mriq_pers_z, mriq_pers_Qr, mriq_pers_Qi,
  //                                                           mriq_grid2_dim_x, mriq_block2_dim_x,
  //                                                           mriq_iter)));
  // cudaErrCheck(cudaEventRecord(stop_kernel));
  // cudaErrCheck(cudaEventSynchronize(stop_kernel));
  // cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
  // printf("[ORI] mriq took %f ms\n\n", kernel_time);

  cudaErrCheck(cudaEventDestroy(start_kernel));
  cudaErrCheck(cudaEventDestroy(stop_kernel));
  cudaErrCheck(cudaDeviceReset());
  return 0;
}
