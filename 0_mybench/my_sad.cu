#include <stdio.h>
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
using namespace nvcuda; // ??? 여기서 에러가 발생합니다. 

#include "header/sad_header.h"
#include "file_t/sad_kernel.cu"


int main(int argc, char* argv[]) {
    int sad_blks = 8;
    int sad_iter = 100;
    if (argc == 3) {
        sad_blks = atoi(argv[1]);
        sad_iter = atoi(argv[2]);
    } 

	float kernel_time;
    // curandGenerator_t gen;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));

    float sub_time;
    cudaEvent_t startSUBKERNEL;
    cudaEvent_t stopSUBKERNEL;
    cudaErrCheck(cudaEventCreate(&startSUBKERNEL));
    cudaErrCheck(cudaEventCreate(&stopSUBKERNEL));

    int image_width_macroblocks = 120;
    int image_height_macroblocks = 67;
    int ref_image_width = 1920;
    int ref_image_height = 1072;
    int image_size_bytes = ref_image_width * ref_image_height * sizeof(short);
    int image_size_macroblocks = image_width_macroblocks * image_height_macroblocks;

    struct cudaArray *ori_ref_ary;  /* Reference image on the device */
    short *ori_cur_image;         /* Current image on the device */
    unsigned short *ori_d_sads;     /* SADs on the device */

	struct cudaArray *ptb_ref_ary;  /* Reference image on the device */
    short *ptb_cur_image;         /* Current image on the device */
    unsigned short *ptb_d_sads;     /* SADs on the device */

	unsigned short *host_ori_d_sads;
	unsigned short *host_ptb_d_sads;

    cudaMalloc((void **)&ori_cur_image, image_size_bytes);
    cudaMallocArray(&ori_ref_ary, &get_ref().channelDesc, ref_image_width, ref_image_height);
    cudaBindTextureToArray(get_ref(), ori_ref_ary);
	cudaMemset(ori_cur_image, 100, image_size_bytes);
    cudaMalloc((void **)&ori_d_sads, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));
    cudaMemset(ori_d_sads, 0, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));

	cudaMalloc((void **)&ptb_cur_image, image_size_bytes);
    cudaMallocArray(&ptb_ref_ary, &get_ref_2().channelDesc, ref_image_width, ref_image_height);
    cudaBindTextureToArray(get_ref_2(), ptb_ref_ary);
	cudaMemset(ptb_cur_image, 100, image_size_bytes);
    cudaMalloc((void **)&ptb_d_sads, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));
    cudaMemset(ptb_d_sads, 0, 41 * MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));

	host_ori_d_sads = (unsigned short *)malloc(MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));
	host_ptb_d_sads = (unsigned short *)malloc(MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short));
    
	printf("[ORI] Running with sad...\n");
	cudaErrCheck(cudaEventRecord(startKERNEL));

	dim3 sad_grid1;
    dim3 sad_block1;
	sad_grid1.x = CEIL(ref_image_width / 4, THREADS_W);
	sad_grid1.y = CEIL(ref_image_height / 4, THREADS_H);
    sad_block1.x = CEIL(MAX_POS, POS_PER_THREAD) * THREADS_W * THREADS_H;
    printf("[ORI] sad_grid1 -- %d * %d * %d sad_block1 -- %d * %d * %d \n", 
        sad_grid1.x, sad_grid1.y, sad_grid1.z, sad_block1.x, sad_block1.y, sad_block1.z);

    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
    checkKernelErrors((ori_mb_sad_calc<<<sad_grid1, sad_block1, SAD_LOC_SIZE_BYTES>>>
      (ori_d_sads, (unsigned short *)ori_cur_image, image_width_macroblocks, image_height_macroblocks)));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    printf("[ORI] sub1 took %f ms\n", sub_time);

	dim3 sad_grid2;
    dim3 sad_block2;
	sad_grid2.x = image_width_macroblocks;
	sad_grid2.y = image_height_macroblocks;
	sad_block2.x = 32;
	sad_block2.y = 4;
	printf("[ORI] sad_grid2 -- %d * %d * %d sad_block2 -- %d * %d * %d \n", 
        sad_grid2.x, sad_grid2.y, sad_grid2.z, sad_block2.x, sad_block2.y, sad_block2.z);

    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
    checkKernelErrors((ori_larger_sad_calc_8<<<sad_grid2, sad_block2>>>
			(ori_d_sads, image_width_macroblocks, image_height_macroblocks)));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    printf("[ORI] sub2 took %f ms\n", sub_time);

	dim3 sad_grid3;
    dim3 sad_block3;
	sad_grid3.x = image_width_macroblocks;
	sad_grid3.y = image_height_macroblocks;
	sad_block3.x = 32;
	sad_block3.y = 1;
    printf("[ORI] sad_grid3 -- %d * %d * %d sad_block3 -- %d * %d * %d \n", 
        sad_grid3.x, sad_grid3.y, sad_grid3.z, sad_block3.x, sad_block3.y, sad_block3.z);

    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
    checkKernelErrors((ori_larger_sad_calc_16<<<sad_grid3, sad_block3>>>
      (ori_d_sads, image_width_macroblocks, image_height_macroblocks)));
	
	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopKERNEL));
    printf("[ORI] sub3 took %f ms\n", sub_time);
    printf("[ORI] sad took %f ms\n\n", kernel_time);

	printf("[PTB] Running with sad...\n");
	cudaErrCheck(cudaEventRecord(startKERNEL));

	// int grid1_dimension_x = sad_grid1.x;
	// int grid1_dimension_y = sad_grid1.y;
	// int block1_dimension_x = sad_block1.x;
	// int block1_dimension_y = sad_block1.y;
    // sad_grid1.x = sad_blks == 0 ? grid1_dimension_x * grid1_dimension_y : 68 * sad_blks;
	// sad_grid1.y = 1;
	// sad_block1.x = block1_dimension_x * block1_dimension_y;
	// sad_block1.y = 1;
	printf("[PTB] sad_grid1 -- %d * %d * %d sad_block1 -- %d * %d * %d \n", 
        sad_grid1.x, sad_grid1.y, sad_grid1.z, sad_block1.x, sad_block1.y, sad_block1.z);

    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
	checkKernelErrors((ori_mb_sad_calc<<<sad_grid1, sad_block1>>>
			(ptb_d_sads, (unsigned short *)ptb_cur_image, image_width_macroblocks, image_height_macroblocks
			// grid1_dimension_x, grid1_dimension_y, block1_dimension_x, block1_dimension_y, sad_iter1
        )));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    printf("[PTB] sub1 took %f ms\n", sub_time);

	// int grid2_dimension_x = sad_grid2.x;
	// int grid2_dimension_y = sad_grid2.y;
	// int block2_dimension_x = sad_block2.x;
	// int block2_dimension_y = sad_block2.y;
	// sad_grid2.x = 68;
	// sad_grid2.y = 1;
	// sad_block2.x = block2_dimension_x * block2_dimension_y;
	// sad_block2.y = 1;
	printf("[PTB] sad_grid2 -- %d * %d * %d sad_block2 -- %d * %d * %d \n", 
        sad_grid2.x, sad_grid2.y, sad_grid2.z, sad_block2.x, sad_block2.y, sad_block2.z);

    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
	checkKernelErrors((ori_larger_sad_calc_8<<<sad_grid2, sad_block2>>>
			(ptb_d_sads, image_width_macroblocks, image_height_macroblocks
			// grid2_dimension_x, grid2_dimension_y, block2_dimension_x, block2_dimension_y, 
            // sad_iter2
        )));
    cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
    printf("[PTB] sub2 took %f ms\n", sub_time);


	// int grid3_dimension_x = sad_grid3.x;
	// int grid3_dimension_y = sad_grid3.y;
	// int block3_dimension_x = sad_block3.x;
	// int block3_dimension_y = sad_block3.y;
	// sad_grid3.x = 68;
	// sad_grid3.y = 1;
	// sad_block3.x = block3_dimension_x * block3_dimension_y;
	// sad_block3.y = 1;
    
    printf("[PTB] sad_grid3 -- %d * %d * %d sad_block3 -- %d * %d * %d \n", 
        sad_grid3.x, sad_grid3.y, sad_grid3.z, sad_block3.x, sad_block3.y, sad_block3.z);
    cudaErrCheck(cudaEventRecord(startSUBKERNEL));
	checkKernelErrors((ori_larger_sad_calc_16<<<sad_grid3, sad_block3>>>
      		(ptb_d_sads, image_width_macroblocks, image_height_macroblocks)));

	cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopKERNEL));
    printf("[PTB] sub3 took %f ms\n", sub_time);
    printf("[PTB] sad took %f ms\n\n", kernel_time);

	// Error checking
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_ori_d_sads, ori_d_sads, MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_ptb_d_sads, ptb_d_sads, MAX_POS_PADDED * image_size_macroblocks * sizeof(unsigned short), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < MAX_POS_PADDED * image_size_macroblocks; i++) {
        unsigned short v1 = host_ori_d_sads[i];
        unsigned short v2 = host_ptb_d_sads[i];
        if (v1 - v2 != 0) {
        errors++;
        if (errors < 10) printf("%hu %hu\n", v1, v2);
        }
    }

    if (errors > 0) {
        printf("ORIGIN VERSION does not agree with MY VERSION! %d errors in %d values!\n", errors, MAX_POS_PADDED * image_size_macroblocks);
    }
    else {
        printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaDeviceReset());
    return 0;
}