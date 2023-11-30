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

#define TILE_WIDTH 16
#define TILE_HEIGHT 16


//global kernal for matrix multiplication, takes in input matrices and sizes, and multiplies them
//matrix multiplication is being done tile by tile
__global__ void matrix_mult(float* MA, float* MB, float* MC, int M, int N, int K)
{
    //shared memory takes one tile at a time
    __shared__ float Sub_A[TILE_WIDTH][TILE_HEIGHT];	//to store tiles for Matrix A
    __shared__ float Sub_B[TILE_HEIGHT][TILE_WIDTH];	//to store tiles for Matrix B

    //threads x and y index for the current block
    unsigned int tx=threadIdx.x;
    unsigned int ty=threadIdx.y;

    unsigned int r=blockIdx.y*blockDim.y + threadIdx.y;	//column value using y-index of current thread
    unsigned int c=blockIdx.x*blockDim.x + threadIdx.x;	//row value using x-index of current thread
    
    unsigned int idx=c*M+r;				//column major index, using row and column value
    
    float val=0;		//register to store multiplication result initialized to zero

    for(int m=0; m<1+((K-1)/TILE_WIDTH);m++)	//going over all tiles one by one, with each m
    {

        int var1=m*TILE_WIDTH+tx ;		//x thread value for current tile
        int var2=m*TILE_WIDTH+ty ;		//y thread value for current tile
        
        //copying a tile from MA
        if (r < M && var1 < K)		//if the value is associated to a valid matrix coordinate in MA then store it to shared memory Sub_A
            Sub_A[ty][tx]=MA[r + var1*M];//storing a "valid" value from array to shared memory
        else
            Sub_A[ty][tx]=0;					//storing zero, since there is no valid value
        __syncthreads();						//syncing all threads once shared memory Sub_A is stored
        
        //copying a tile from MB
        if(c < N && var2 < K)	//if value is associates to a valid matrix coordinate in MB then store it to shared memory Sub_B
            Sub_B[ty][tx]=MB[var2+K*c];	//storing the valid value
        else 
            Sub_B[ty][tx]=0;		//storing zero, since no valid value
        __syncthreads();		//synchronizing threads
        

        for(int i=0; i<TILE_WIDTH;i++)	//going over entire tile, ty row in Sub_A and tx column in Sub_B
            val+=Sub_A[ty][i]*Sub_B[i][tx];	//and multiplying elements
        __syncthreads();		//synchronizing threads
    }
    
    //removing degenerate cases
    if(r < M && c< N) {
        MC[idx]=val;	//saving multiplication result to global memory
    }
        
}

int main(int argc, char* argv[]) {

    // variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));
    cublasHandle_t cublasHandle;
	cublasErrCheck(cublasCreate(&cublasHandle));

    // sgemm variables
    // ---------------------------------------------------------------------------------------
    float *sgemm_ori_a;
    float *sgemm_ori_b;
    float *sgemm_ori_c;
    float *sgemm_ptb_a;
    float *sgemm_ptb_b;
    float *sgemm_ptb_c;
    float *host_sgemm_ori_c;
    float *host_sgemm_ptb_c;

    // parallel experiment
    int NORMAL_M = 1024;
    int NORMAL_N = 1024;
    int NORMAL_K = 1024;

    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_a, NORMAL_M * NORMAL_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_b, NORMAL_K * NORMAL_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void**)&sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float)));

    host_sgemm_ori_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));
    host_sgemm_ptb_c = (float *)malloc(NORMAL_M * NORMAL_N * sizeof(float));

    curandGenerator_t gen;
    curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    curandErrCheck(curandGenerateUniform(gen, sgemm_ori_a, NORMAL_M * NORMAL_K));
    curandErrCheck(curandGenerateUniform(gen, sgemm_ori_b, NORMAL_K * NORMAL_N));
    cudaErrCheck(cudaMemcpy(sgemm_ptb_a, sgemm_ori_a, NORMAL_M * NORMAL_K * sizeof(float), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(sgemm_ptb_b, sgemm_ori_b, NORMAL_K * NORMAL_N * sizeof(float), cudaMemcpyDeviceToDevice));
    curandErrCheck(curandDestroyGenerator(gen));

    // SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 sgemm_grid;
    dim3 sgemm_block;
    sgemm_block.x = TILE_WIDTH;
    sgemm_block.y = TILE_HEIGHT;
    sgemm_grid.x = NORMAL_N/TILE_WIDTH;
    sgemm_grid.y = NORMAL_M/TILE_HEIGHT;
    printf("[ORI] Running with sgemm...\n");
    printf("[ORI] sgemm_grid -- %d * %d sgemm_block -- %d * %d \n", 
        sgemm_grid.x, sgemm_grid.y, sgemm_block.x, sgemm_block.y);

    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((matrix_mult <<< sgemm_grid, sgemm_block >>> (sgemm_ori_a, sgemm_ori_b, sgemm_ori_c, 
                            NORMAL_M, NORMAL_N, NORMAL_K)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[ORI] sgemm took %f ms\n\n", kernel_time);

    float alpha = 1.0;
	float beta = 0.0;
    // PTB running
    // ---------------------------------------------------------------------------------------
    cudaErrCheck(cudaEventRecord(startKERNEL));
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, NORMAL_M, NORMAL_N, NORMAL_K, &alpha, sgemm_ptb_a, NORMAL_M, sgemm_ptb_b, NORMAL_K, &beta, sgemm_ptb_c, NORMAL_M);
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[CUBLAS] sgemm took %f ms\n\n", kernel_time);

    // Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_sgemm_ori_c, sgemm_ori_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_sgemm_ptb_c, sgemm_ptb_c, NORMAL_M * NORMAL_N * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < NORMAL_M * NORMAL_N; i++) {
        float v1 = host_sgemm_ori_c[i];
        float v2 = host_sgemm_ptb_c[i];
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

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaFree(sgemm_ori_a));
    cudaErrCheck(cudaFree(sgemm_ori_b));
    cudaErrCheck(cudaFree(sgemm_ori_c));
    cudaErrCheck(cudaFree(sgemm_ptb_a));
    cudaErrCheck(cudaFree(sgemm_ptb_b));
    cudaErrCheck(cudaFree(sgemm_ptb_c));

    free(host_sgemm_ori_c);
    free(host_sgemm_ptb_c);

    cudaErrCheck(cudaDeviceReset());
    return 0;
}