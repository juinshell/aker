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
using namespace nvcuda;


#include "header/histo_header.h"
#include "file_t/histo_kernel.cu"


int main(int argc, char* argv[]) {
    int histo_blks = 1;
    int histo_iter = 1;
    if (argc == 3) {
        histo_blks = atoi(argv[1]);
        histo_iter = atoi(argv[2]);
    } 

    // variables
    // ---------------------------------------------------------------------------------------
        float kernel_time;
        cudaEvent_t startKERNEL;
        cudaEvent_t stopKERNEL;
        cudaErrCheck(cudaEventCreate(&startKERNEL));
        cudaErrCheck(cudaEventCreate(&stopKERNEL));

        float sub_time;
        cudaEvent_t startSUBKERNEL;
        cudaEvent_t stopSUBKERNEL;
        cudaErrCheck(cudaEventCreate(&startSUBKERNEL));
        cudaErrCheck(cudaEventCreate(&stopSUBKERNEL));
    // ---------------------------------------------------------------------------------------

    // histo variables
    // ---------------------------------------------------------------------------------------
        char inpFiles[] = "./header/histo_img.bin";
        unsigned int img_width, img_height;
        unsigned int histo_width, histo_height;

        FILE* f = fopen(inpFiles, "rb");
        int result = 0;
        result += fread(&img_width,    sizeof(unsigned int), 1, f);
        result += fread(&img_height,   sizeof(unsigned int), 1, f);
        result += fread(&histo_width,  sizeof(unsigned int), 1, f);
        result += fread(&histo_height, sizeof(unsigned int), 1, f);

        printf("img_height %d img_width %d \n", img_height, img_width);
        printf("histo_height %d histo_width %d \n", histo_height, histo_width);

        unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
        result = fread(img, sizeof(unsigned int), img_width*img_height, f);
        fclose(f);
        if (result != img_width*img_height){
            fputs("[Error] Reading input array from file\n", stderr);
            return -1;
        }

        int even_width = ((img_width+1)/2)*2;
        unsigned int* ori_input;
        unsigned int* ori_ranges;
        uchar4* ori_sm_mappings;
        unsigned int* ori_global_subhisto;
        unsigned short* ori_global_histo;
        unsigned int* ori_global_overflow;
        unsigned char* ori_final_histo;

        uchar4* ptb_sm_mappings;
        unsigned int* ptb_global_subhisto;
        unsigned short* ptb_global_histo;
        unsigned int* ptb_global_overflow;

        unsigned short* host_ori_subhisto;
        unsigned short* host_ptb_subhisto;

        cudaMalloc((void**)&ori_input           , 4 * even_width*(((img_height+UNROLL-1)/UNROLL)*UNROLL)*sizeof(unsigned int));
        cudaMalloc((void**)&ori_ranges          , 2*sizeof(unsigned int));
        cudaMalloc((void**)&ori_sm_mappings     , img_width*img_height*sizeof(uchar4));
        cudaMalloc((void**)&ori_global_subhisto , BLOCK_X*img_width*histo_height*sizeof(unsigned int));
        cudaMalloc((void**)&ori_global_histo    , img_width*histo_height*sizeof(unsigned short));
        cudaMalloc((void**)&ori_global_overflow , img_width*histo_height*sizeof(unsigned int));
        cudaMalloc((void**)&ori_final_histo     , img_width*histo_height*sizeof(unsigned char));

        cudaMalloc((void**)&ptb_sm_mappings     , img_width*img_height*sizeof(uchar4));
        cudaMalloc((void**)&ptb_global_subhisto , BLOCK_X*img_width*histo_height*sizeof(unsigned int));
        cudaMalloc((void**)&ptb_global_histo    , img_width*histo_height*sizeof(unsigned short));
        cudaMalloc((void**)&ptb_global_overflow , img_width*histo_height*sizeof(unsigned int));

        host_ori_subhisto = (unsigned short *)calloc(img_width*histo_height, sizeof(unsigned short));
        host_ptb_subhisto = (unsigned short *)calloc(img_width*histo_height, sizeof(unsigned short));

        for (int y=0; y < img_height; y++){
            for (int i = 0; i < 4; i++) {
                cudaMemcpy(&(((unsigned int*)ori_input)[(4*y+i)*even_width]),&img[y*img_width],img_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
            }
        }
        even_width = 4 * even_width;
    // ---------------------------------------------------------------------------------------


    // PRE running
    // ---------------------------------------------------------------------------------------
        unsigned int ranges_h[2] = {UINT32_MAX, 0};
        dim3 gridDim1, blockDim1;
        gridDim1.x = PRESCAN_BLOCKS_X;
        blockDim1.x = PRESCAN_THREADS;
        // printf("[ORI][histo 1] gridDim1 -- %d * %d blockDim1 -- %d * %d \n", gridDim1.x, gridDim1.y, blockDim1.x, blockDim1.y);
        cudaMemcpy(ori_ranges, ranges_h, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaErrCheck(cudaEventRecord(startSUBKERNEL));
        checkKernelErrors((histo_prescan_kernel<<<gridDim1, blockDim1>>>(
                (unsigned int*)ori_input, img_height*img_width, ori_ranges)));
        cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
        // printf("[ORI] sub took %f ms\n", sub_time);
        
        dim3 gridDim2, blockDim2;
        gridDim2.x = (img_height + UNROLL-1)/UNROLL;
        blockDim2.x = (img_width+1)/2;
        // printf("[ORI][histo 2] gridDim2 -- %d * %d blockDim2 -- %d * %d \n", gridDim2.x, gridDim2.y, blockDim2.x, blockDim2.y);
        cudaMemcpy(ranges_h,ori_ranges, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemset(ori_global_subhisto, 0, img_width*histo_height*sizeof(unsigned int));
        cudaErrCheck(cudaEventRecord(startSUBKERNEL));
        checkKernelErrors((histo_intermediates_kernel<<<gridDim2, blockDim2>>>(
                (uint2*)(ori_input), (unsigned int)img_height, (unsigned int)img_width,
                (img_width+1)/2, (uchar4*)(ori_sm_mappings))));
        cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
        // printf("[ORI] sub took %f ms\n", sub_time);

        cudaMemset(ptb_global_subhisto, 0, img_width*histo_height*sizeof(unsigned int));
        cudaMemcpy(ptb_sm_mappings, ori_sm_mappings, img_width*img_height*sizeof(uchar4), cudaMemcpyDeviceToDevice);
    // ---------------------------------------------------------------------------------------


    // SOLO running
    // ---------------------------------------------------------------------------------------
        dim3 gridDim3, blockDim3;
        gridDim3.x = BLOCK_X;
        gridDim3.y = ranges_h[1]-ranges_h[0] + 1;
        blockDim3.x = THREADS;
        printf("[ORI] Running with histo...\n");
        printf("[ORI] gridDim3 -- %d * %d blockDim3 -- %d * %d \n", gridDim3.x, gridDim3.y, blockDim3.x, blockDim3.y);
        cudaErrCheck(cudaEventRecord(startSUBKERNEL));
        checkKernelErrors((ori_histo<<<gridDim3, blockDim3>>>(
                (uchar4*)(ori_sm_mappings), img_height*img_width, ranges_h[0], ranges_h[1],
                histo_height, histo_width, (unsigned int*)(ori_global_subhisto), 
                (unsigned int*)(ori_global_histo), (unsigned int*)(ori_global_overflow), histo_iter)));
        cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
        printf("[ORI] histo took %f ms\n", sub_time);
    // ---------------------------------------------------------------------------------------


    // TMP CODE
    // ---------------------------------------------------------------------------------------
        // dim3 gridDim4, blockDim4;
        // gridDim4.x = BLOCK_X * 3;
        // blockDim4.x = 512;
        // // printf("[ORI][histo 4] gridDim4 -- %d * %d blockDim4 -- %d * %d \n", gridDim4.x, gridDim4.y, blockDim4.x, blockDim4.y);
        // cudaErrCheck(cudaEventRecord(startSUBKERNEL));
        // checkKernelErrors((histo_final_kernel<<<gridDim4, blockDim4>>>(
        //             ranges_h[0], ranges_h[1],
        //             histo_height, histo_width,
        //             (unsigned int*)(ori_global_subhisto),
        //             (unsigned int*)(ori_global_histo),
        //             (unsigned int*)(ori_global_overflow),
        //             (unsigned int*)(ori_final_histo))));
        // cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
        // cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
        // cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
        // // printf("[ORI] sub took %f ms\n", sub_time);
    // ---------------------------------------------------------------------------------------


    // PTB running
    // ---------------------------------------------------------------------------------------
        printf("[PTB] Running with histo...\n");
        gridDim3.x = BLOCK_X;
        gridDim3.y = ranges_h[1]-ranges_h[0] + 1;
        blockDim3.x = THREADS;
        int grid3_dimension_x = gridDim3.x;
        int grid3_dimension_y = gridDim3.y;
        int block3_dimension_x = blockDim3.x;
        int block3_dimension_y = blockDim3.y;
        gridDim3.x = histo_blks == 0 ? grid3_dimension_x * grid3_dimension_y : 68 * histo_blks;
        gridDim3.y = 1;
        blockDim3.x = block3_dimension_x * block3_dimension_y;
        blockDim3.y = 1;
        printf("[PTB] gridDim3 -- %d * %d blockDim3 -- %d * %d \n", gridDim3.x, gridDim3.y, blockDim3.x, blockDim3.y);
        cudaErrCheck(cudaEventRecord(startSUBKERNEL));
        checkKernelErrors((ptb_histo<<<gridDim3, blockDim3>>>(
                (uchar4*)(ptb_sm_mappings), img_height*img_width, ranges_h[0], ranges_h[1],
                histo_height, histo_width, (unsigned int*)(ptb_global_subhisto),
                (unsigned int*)(ptb_global_histo), (unsigned int*)(ptb_global_overflow),
                grid3_dimension_x, grid3_dimension_y, block3_dimension_x, block3_dimension_y,
                histo_iter)));
        cudaErrCheck(cudaEventRecord(stopSUBKERNEL));
        cudaErrCheck(cudaEventSynchronize(stopSUBKERNEL));
        cudaErrCheck(cudaEventElapsedTime(&sub_time, startSUBKERNEL, stopSUBKERNEL));
        printf("[PTB] histo took %f ms\n", sub_time);
    // ---------------------------------------------------------------------------------------


    // Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_ori_subhisto, ori_global_histo, img_width*histo_height*sizeof(unsigned short), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_ptb_subhisto, ptb_global_histo, img_width*histo_height*sizeof(unsigned short), cudaMemcpyDeviceToHost));

    int errors = 0;
    // int j = 0;
    // for (int i = 0; i < img_width*histo_height; i++) {
    //     unsigned short v1 = (unsigned int)host_ori_subhisto[i];
    //     unsigned short v2 = (unsigned int)host_ptb_subhisto[i];
    //     if (v1 != v2) {
    //         errors++;
    //         if (errors < 10) printf("%hu %hu\n", v1, v2);
    //     }
    //     if (v1 != 0 && j < 3) {
    //         printf("%hu %hu\n", v1, v2);
    //         j++;
    //     }
    // }
    // printf("\n");
    if (errors > 0) {
        printf("ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaDeviceReset());
    return 0;
}