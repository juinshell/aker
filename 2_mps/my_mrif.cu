#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>
#include "switch.h"
#include <cuda_runtime.h>
#include "header/mrif_header.h"
#include "file_t/mrif_kernel.cu"

#include "cuda_ipc.h"
// Define some error checking macros.
#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }
void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
    if (stat != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define checkKernelErrors(expr)                                   \
    do                                                            \
    {                                                             \
        expr;                                                     \
                                                                  \
        cudaError_t __err = cudaGetLastError();                   \
        if (__err != cudaSuccess)                                 \
        {                                                         \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
                   cudaGetErrorString(__err));                    \
            abort();                                              \
        }                                                         \
    } while (0)

int main(int argc, char *argv[])
{
    int mrif_blks = 1;
    int mrif_iter = 2000;
    if (argc == 3)
    {
        mrif_blks = atoi(argv[1]);
        mrif_iter = atoi(argv[2]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    cudaEvent_t start_kernel;
    cudaEvent_t stop_kernel;
    cudaErrCheck(cudaEventCreate(&start_kernel));
    cudaErrCheck(cudaEventCreate(&stop_kernel));

    // mrif variables
    // ---------------------------------------------------------------------------------------
    int numX, numK;                           /* Number of X and K values */
    int original_numK;                        /* Number of K values in input file */
    float *base_kx, *base_ky, *base_kz;       /* K trajectory (3D vectors) */
    float *base_x, *base_y, *base_z;          /* X coordinates (3D vectors) */
    float *base_phiR, *base_phiI;             /* Phi values (complex) */
    float *base_dR, *base_dI;                 /* D values (complex) */
    float *base_realRhoPhi, *base_imagRhoPhi; /* RhoPhi values (complex) */
    kValues *kVals;                           /* Copy of X and RhoPhi.  Its
                                        * data layout has better cache
                                        * performance. */
    inputData(
        &original_numK, &numX,
        &base_kx, &base_ky, &base_kz,
        &base_x, &base_y, &base_z,
        &base_phiR, &base_phiI,
        &base_dR, &base_dI);
    numK = original_numK;

    // createDataStructs(numK, numX, base_realRhoPhi, base_imagRhoPhi, base_outR, base_outI);
    kVals = (kValues *)calloc(numK, sizeof(kValues));

    // kernel 1
    float *ori_phiR, *ori_phiI;
    float *ori_dR, *ori_dI;
    float *ori_realRhoPhi, *ori_imagRhoPhi;
    // kernel 2
    float *ori_x, *ori_y, *ori_z;
    float *ori_outI, *ori_outR;
    float *host_ori_outI; /* Output signal (complex) */

    // kernel 2
    float *pers_x, *pers_y, *pers_z;
    float *pers_outI, *pers_outR;
    float *host_pers_outI; /* Output signal (complex) */

    {
        cudaErrCheck(cudaMalloc((void **)&ori_phiR, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_phiI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dR, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_dI, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_realRhoPhi, numK * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_imagRhoPhi, numK * sizeof(float)));
        // host_ori_phiMag = (float* ) memalign(16, numK * sizeof(float));
        cudaErrCheck(cudaMemcpy(ori_phiR, base_phiR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_phiI, base_phiI, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dR, base_dR, numK * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_dI, base_dI, numK * sizeof(float), cudaMemcpyHostToDevice));

        base_realRhoPhi = (float *)calloc(numK, sizeof(float));
        base_imagRhoPhi = (float *)calloc(numK, sizeof(float));

        cudaErrCheck(cudaMalloc((void **)&ori_x, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_y, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_z, numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(ori_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_x, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(ori_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&ori_outR, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&ori_outI, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outR, 0, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(ori_outI, 0, numX * sizeof(float)));

        cudaErrCheck(cudaMalloc((void **)&pers_x, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&pers_y, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&pers_z, numX * sizeof(float)));
        cudaErrCheck(cudaMemcpy(pers_x, base_x, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(pers_x, base_y, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(pers_z, base_z, numX * sizeof(float), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMalloc((void **)&pers_outR, numX * sizeof(float)));
        cudaErrCheck(cudaMalloc((void **)&pers_outI, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(pers_outR, 0, numX * sizeof(float)));
        cudaErrCheck(cudaMemset(pers_outI, 0, numX * sizeof(float)));

        host_ori_outI = (float *)calloc(numX, sizeof(float));
        host_pers_outI = (float *)calloc(numX, sizeof(float));
    }

    // mrif kernel 1
    // ---------------------------------------------------------------------------------------
    // computeRhoPhi_GPU(numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi);
    dim3 mrif_grid1;
    dim3 mrif_block1;
    mrif_grid1.x = numK / KERNEL_RHO_PHI_THREADS_PER_BLOCK;
    mrif_grid1.y = 1;
    mrif_block1.x = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
    mrif_block1.y = 1;
    printf("[ORI] mrif_grid1 -- %d * %d * %d mrif_block1 -- %d * %d * %d \n",
           mrif_grid1.x, mrif_grid1.y, mrif_grid1.z, mrif_block1.x, mrif_block1.y, mrif_block1.z);
    checkKernelErrors((ComputeRhoPhiGPU<<<mrif_grid1, mrif_block1>>>(numK, ori_phiR, ori_phiI, ori_dR, ori_dI, ori_realRhoPhi, ori_imagRhoPhi)));
    cudaErrCheck(cudaMemcpy(base_realRhoPhi, ori_realRhoPhi, numK * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(base_imagRhoPhi, ori_imagRhoPhi, numK * sizeof(float), cudaMemcpyDeviceToHost));

    for (int k = 0; k < numK; k++)
    {
        kVals[k].Kx = base_kx[k];
        kVals[k].Ky = base_ky[k];
        kVals[k].Kz = base_kz[k];
        kVals[k].RhoPhiR = base_realRhoPhi[k];
        kVals[k].RhoPhiI = base_imagRhoPhi[k];
    }

    // computeFH_GPU(numK, numX, x_d, y_d, z_d, kVals, outR_d, outI_d);

    // SOLO running
    // ---------------------------------------------------------------------------------------
    int FHGrids = numK / KERNEL_FH_K_ELEMS_PER_GRID;
    dim3 mrif_grid2;
    dim3 mrif_block2;
    mrif_grid2.x = numX / KERNEL_FH_THREADS_PER_BLOCK;
    mrif_grid2.y = 1;
    mrif_block2.x = KERNEL_FH_THREADS_PER_BLOCK;
    mrif_block2.y = 1;
    printf("[ORI] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \n",
           mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);
    // printf("FHGrids %d \n", FHGrids);

    int FHGridBase = 0 * KERNEL_FH_K_ELEMS_PER_GRID;
    kValues *kValsTile = kVals + FHGridBase;
    int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, numK - FHGridBase);
    cudaMemcpyToSymbol(c, kValsTile, numElems * sizeof(kValues), 0);

    /* Barrier */
    int fd = shm_open(MMAP_FILE, O_RDWR, 0666);
    // ftruncate(fd, sizeof(pthread_barrier_t));

    pthread_barrier_t *shared_barrier = (pthread_barrier_t *)mmap(NULL, sizeof(pthread_barrier_t), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    pthread_barrier_wait(shared_barrier);

    cudaErrCheck(cudaEventRecord(start_kernel));
    checkKernelErrors((ori_ComputeFH<<<mrif_grid2, mrif_block2>>>(numK, FHGridBase, ori_x, ori_y, ori_z, ori_outR, ori_outI, mrif_iter)));
    cudaErrCheck(cudaEventRecord(stop_kernel));
    cudaErrCheck(cudaEventSynchronize(stop_kernel));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
    printf("[ORI] mrif took %f ms\n\n", kernel_time);

    // PTB running
    // ---------------------------------------------------------------------------------------
    int mrif_grid2_dim_x = mrif_grid2.x;
    int mrif_block2_dim_x = mrif_block2.x;
    mrif_grid2.x = mrif_blks == 0 ? mrif_grid2_dim_x : 68 * mrif_blks;
    printf("[PERS] Running with mrif...\n");
    printf("[PERS] mrif_grid2 -- %d * %d * %d mrif_block2 -- %d * %d * %d \n",
           mrif_grid2.x, mrif_grid2.y, mrif_grid2.z, mrif_block2.x, mrif_block2.y, mrif_block2.z);

    pthread_barrier_wait(shared_barrier);
    cudaErrCheck(cudaEventRecord(start_kernel));
    checkKernelErrors((pers_ComputeFH<<<mrif_grid2, mrif_block2>>>(numK, FHGridBase, pers_x, pers_y, pers_z, pers_outR, pers_outI,
                                                                   mrif_grid2_dim_x, mrif_block2_dim_x,
                                                                   mrif_iter)));
    cudaErrCheck(cudaEventRecord(stop_kernel));
    cudaErrCheck(cudaEventSynchronize(stop_kernel));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
    printf("[PERS] mrif took %f ms\n\n", kernel_time);

    // cudaErrCheck(cudaEventRecord(start_kernel));
    // checkKernelErrors((pers_ComputeFH<<<mrif_grid2, mrif_block2>>>(numK, FHGridBase, pers_x, pers_y, pers_z, pers_outR, pers_outI,
    //                                                                mrif_grid2_dim_x, mrif_block2_dim_x,
    //                                                                mrif_iter)));
    // cudaErrCheck(cudaEventRecord(stop_kernel));
    // cudaErrCheck(cudaEventSynchronize(stop_kernel));
    // cudaErrCheck(cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel));
    // printf("[PERS] mrif took %f ms\n\n", kernel_time);

    cudaErrCheck(cudaEventDestroy(start_kernel));
    cudaErrCheck(cudaEventDestroy(stop_kernel));
    cudaErrCheck(cudaDeviceReset());
    return 0;
}
