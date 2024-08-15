#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <deque>
#include <iostream>

#include <assert.h>
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

#include "header/tzgemm_header.h"
// #include "header/tzgemm_ori.h"
#include "header/bfs_header.h"

#include "file_t/tzgemm_kernel.cu"
// #include "file_t/tzgemm_ori.cu"
#include "file_t/bfs_kernel.cu"


__global__ void mix_kernel(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
	int *q1,
    int *q2, 
	Node *g_graph_nodes, 
	Edge *g_graph_edges, 
	int *g_color, 
	int *g_cost, 
	int no_of_nodes, 
	int *tail, 
	int gray_shade, 
	int k,
	int *overflow,
	int grid_dimension_x,
	int block_dimension_x,
	int bfs_iter){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < BFS_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_bfs(q1, q2, 
           g_graph_nodes, g_graph_edges, 
           g_color, g_cost, 
           no_of_nodes, 
           tail, 
           gray_shade, 
           k,
           overflow,
           grid_dimension_x,
           block_dimension_x,
		   thread_step,
           bfs_iter
           );
    }
}

__global__ void tgemm_bfs_kernel(
    half *a, half *b, float *c,
    int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
	int *q1,
    int *q2, 
	Node *g_graph_nodes, 
	Edge *g_graph_edges, 
	int *g_color, 
	int *g_cost, 
	int no_of_nodes, 
	int *tail, 
	int gray_shade, 
	int k,
	int *overflow,
	int grid_dimension_x,
	int block_dimension_x,
	int bfs_iter){
    if (threadIdx.x < wmma_block_dim_x * 1 && blockIdx.x < WMMA_GRID_DIM2) {
        mix_tzgemm0(a, b, c, MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter);
    } else if (threadIdx.x >= wmma_block_dim_x * 1 && blockIdx.x < BFS_GRID_DIM) {
        int thread_step = wmma_block_dim_x * 1;
        mix_bfs(q1, q2, 
           g_graph_nodes, g_graph_edges, 
           g_color, g_cost, 
           no_of_nodes, 
           tail, 
           gray_shade, 
           k,
           overflow,
           grid_dimension_x,
           block_dimension_x,
		   thread_step,
           bfs_iter
           );
    }
}

#include "./fspmv.h"

int main(int argc, char* argv[]) {
    int bfs_blks = 1;
	int bfs_iter = 	5000;
	int wmma_blks = 2;
    int wmma_iter = 1900;
    int M_INPUT = 128 * 1;
	int N_INPUT = 128 * 3136;
	int K_INPUT = 128 * 1;
	int mixwarp = 3;
	if (argc == 2) {
		mixwarp = atoi(argv[1]);
	} else if (argc == 4) {
        bfs_blks = atoi(argv[1]);
        bfs_iter = atoi(argv[2]);
		mixwarp = atoi(argv[3]);
    }

    // variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    float serial_time = 0;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));
	cudaStream_t streams[2];
    for (int i = 0; i < 2; i++) {
        cudaErrCheck(cudaStreamCreate(&streams[i]));
    }


    // tcgemm variables
    // ---------------------------------------------------------------------------------------
		int MATRIX_M = (M_INPUT < 64) ? 64 : (M_INPUT / 64) * 64;
		int MATRIX_N = (N_INPUT < 64) ? 64 : (N_INPUT / 64) * 64;
		int MATRIX_K = (K_INPUT < 64) ? 64 : (K_INPUT / 64) * 64;

		int M_TILES = MATRIX_M / WMMA_M;
		int N_TILES = MATRIX_N / WMMA_N;
		int K_TILES = MATRIX_K / WMMA_K;

		printf("M_ORI: %5d MATRIX_M: %5d (%d x %d) \n", M_INPUT, MATRIX_M, WMMA_M, M_TILES);
		printf("N_ORI: %5d MATRIX_N: %5d (%d x %d) \n", N_INPUT, MATRIX_N, WMMA_N, N_TILES);
		printf("K_ORI: %5d MATRIX_K: %5d (%d x %d) \n", K_INPUT, MATRIX_K, WMMA_K, K_TILES);

		float *ori_host_A = NULL;
		float *ori_host_B = NULL;
		float *host_wmma_ori_c = NULL;
		float *host_wmma_ptb_c = NULL;

		half *wmma_ori_a = NULL;
		half *wmma_ori_b = NULL;
		float *wmma_ori_c = NULL;
		float *wmma_ptb_c = NULL;

		host_wmma_ori_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);
		host_wmma_ptb_c = (float *)malloc(sizeof(float) * MATRIX_M * MATRIX_N);

		cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_A), sizeof(float) * MATRIX_M * MATRIX_K));
		cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&ori_host_B), sizeof(float) * MATRIX_N * MATRIX_K));
		cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_a), sizeof(half) * MATRIX_M * MATRIX_K));
		cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_b), sizeof(half) * MATRIX_N * MATRIX_K));
		cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ori_c), sizeof(float) * MATRIX_M * MATRIX_N));
		cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&wmma_ptb_c), sizeof(float) * MATRIX_M * MATRIX_N));

		assert(((unsigned long long)wmma_ori_a) % 128 == 0);
		assert(((unsigned long long)wmma_ori_b) % 128 == 0);
		assert(((unsigned long long)wmma_ori_c) % 128 == 0);
		assert(((unsigned long long)wmma_ptb_c) % 128 == 0);

		curandGenerator_t gen;
		curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
		curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
		curandErrCheck(curandGenerateUniform(gen, ori_host_A, MATRIX_M * MATRIX_K));
		curandErrCheck(curandGenerateUniform(gen, ori_host_B, MATRIX_N * MATRIX_K));
		convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_a, ori_host_A, MATRIX_M * MATRIX_K);
		convertFp32ToFp16 <<< (MATRIX_N * MATRIX_K + 255) / 256, 256 >>> (wmma_ori_b, ori_host_B, MATRIX_N * MATRIX_K);
		cudaErrCheck(cudaMemset(wmma_ori_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
		cudaErrCheck(cudaMemset(wmma_ptb_c, 0, sizeof(float) * MATRIX_M * MATRIX_N));
    // ---------------------------------------------------------------------------------------


    // bfs variables
    // ---------------------------------------------------------------------------------------
	//the number of nodes in the graph
	int num_of_nodes = 0; 
	//the number of edges in the graph
	int num_of_edges = 0;

	// Read in Graph from a file
	// ------------------------------------------------------------------
		int source;
		int start, edgeno;

		char file_dat[] = "../0_mybench/file_t/bfs_input.dat";
		fp = fopen(file_dat,"r");
		if(!fp) {
			printf("Error Reading graph file\n");
			return 0;
		}
	
		fscanf(fp,"%d",&num_of_nodes);
		// allocate host memory and initialize
		Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*num_of_nodes);
		int *host_bfs_bfs_ori_color = (int*) malloc(sizeof(int)*num_of_nodes);
		int *host_bfs_bfs_ptb_color = (int*) malloc(sizeof(int)*num_of_nodes);
		for( unsigned int i = 0; i < num_of_nodes; i++) {
			fscanf(fp,"%d %d",&start,&edgeno);
			h_graph_nodes[i].x = start;
			h_graph_nodes[i].y = edgeno;
			host_bfs_bfs_ori_color[i] = WHITE;
			host_bfs_bfs_ptb_color[i] = WHITE;
		}

		//read the source node from the file
		fscanf(fp,"%d",&source);
		fscanf(fp,"%d",&num_of_edges);
		int id,cost;
		Edge* h_graph_edges = (Edge*) malloc(sizeof(Edge)*num_of_edges);
		for(int i=0; i < num_of_edges ; i++) {
			fscanf(fp,"%d",&id);
			fscanf(fp,"%d",&cost);
			h_graph_edges[i].x = id;
			h_graph_edges[i].y = cost;
		}

		if(fp){
			fclose(fp);
		}    

		// allocate mem for the result on host side
		int* host_bfs_ori_cost = (int*) malloc(sizeof(int) * num_of_nodes);
		int* host_bfs_ptb_cost = (int*) malloc(sizeof(int) * num_of_nodes);
		for(int i = 0; i < num_of_nodes; i++) {
			host_bfs_ori_cost[i] = INF;
			host_bfs_ptb_cost[i] = INF;
		}
		host_bfs_ori_cost[source] = 0;
		host_bfs_ptb_cost[source] = 0;
	// ------------------------------------------------------------------

	// allocate memory of original version
    // ---------------------------------------------------------------------------------------
		//Copy the Node list to device memory
		Node *bfs_ori_graph_nodes;
		//Copy the Edge List to device Memory
		Edge *bfs_ori_graph_edges;
		int *bfs_ori_color;
		int *bfs_ori_cost;
		int *bfs_ori_q1;
		int *bfs_ori_q2;
		int *bfs_ori_tail;
		int *bfs_ori_front_cont;
		//max number of frontier nodes assigned to a block
		int *bfs_ori_max_nodes_per_block;
		int *bfs_ori_global_kt;
		//whether or not to adjust "k", see comment on "BFS_kernel_multi_blk_inGPU" for more details 
		int *bfs_ori_switch_k;
		int *bfs_ori_num_t;//number of threads
		//whether to stay within a kernel, used in "BFS_kernel_multi_blk_inGPU"
		bool *bfs_ori_stay;

		cudaMalloc((void**) &bfs_ori_graph_edges, sizeof(Edge)*num_of_edges);
		cudaMalloc((void**) &bfs_ori_graph_nodes, sizeof(Node)*num_of_nodes);
		cudaMemcpy(bfs_ori_graph_edges, h_graph_edges, sizeof(Edge)*num_of_edges, cudaMemcpyHostToDevice);
		cudaMemcpy(bfs_ori_graph_nodes, h_graph_nodes, sizeof(Node)*num_of_nodes, cudaMemcpyHostToDevice);

		cudaMalloc((void**) &bfs_ori_color, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ori_cost, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ori_q1, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ori_q2, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ori_tail, sizeof(int));
		cudaMalloc((void**) &bfs_ori_front_cont, sizeof(int));
		cudaMemcpy(bfs_ori_color, host_bfs_bfs_ori_color, sizeof(int)*num_of_nodes, cudaMemcpyHostToDevice);
		cudaMemcpy(bfs_ori_cost, host_bfs_ori_cost, sizeof(int)*num_of_nodes, cudaMemcpyHostToDevice);

		cudaMalloc((void**) &bfs_ori_max_nodes_per_block, sizeof(int));
		cudaMalloc((void**) &bfs_ori_global_kt, sizeof(int));
		cudaMemcpy(bfs_ori_global_kt,&zero, sizeof(int),cudaMemcpyHostToDevice);

		//bind the texture memory with global memory
		cudaBindTexture(0, bfs_ori_graph_node_ref, bfs_ori_graph_nodes, sizeof(Node)*num_of_nodes);
		cudaBindTexture(0, bfs_ori_graph_edge_ref, bfs_ori_graph_edges, sizeof(Edge)*num_of_edges);

		cudaMalloc((void**) &bfs_ori_switch_k, sizeof(int));
		cudaMalloc((void**) &bfs_ori_num_t, sizeof(int));
		cudaMalloc( (void**) &bfs_ori_stay, sizeof(bool));

		cudaMemcpy(bfs_ori_tail, &h_top, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&bfs_ori_cost[source], &zero, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&bfs_ori_q1[0], &source, sizeof(int), cudaMemcpyHostToDevice);

		int host_ori_overflow = 0;
		int *ori_overflow;
		cudaMalloc((void**) &ori_overflow, sizeof(int));
		cudaMemcpy(ori_overflow, &host_ori_overflow, sizeof(int), cudaMemcpyHostToDevice);

		int * switch_kd;
		cudaMalloc((void**) &switch_kd, sizeof(int));
		int * num_td;//number of threads
		cudaMalloc((void**) &num_td, sizeof(int));
		//max number of frontier nodes assigned to a block
		int * max_nodes_per_block_d;
		cudaMalloc( (void**) &max_nodes_per_block_d, sizeof(int));
		int *global_kt_d;
		cudaMalloc( (void**) &global_kt_d, sizeof(int));
		cudaMemcpy(global_kt_d,&zero, sizeof(int),cudaMemcpyHostToDevice);
	// ---------------------------------------------------------------------------------------

	// allocate memory on PTB version
	// ---------------------------------------------------------------------------------------
		//Copy the Node list to device memory
		Node *bfs_ptb_graph_nodes;
		//Copy the Edge List to device Memory
		Edge *bfs_ptb_graph_edges;
		int *bfs_ptb_color;
		int *bfs_ptb_cost;
		int *bfs_ptb_q1;
		int *bfs_ptb_q2;
		int *bfs_ptb_tail;
		int *bfs_ptb_front_cont;
		//max number of frontier nodes assigned to a block
		int *bfs_ptb_max_nodes_per_block;
		int *bfs_ptb_global_kt;
		//whether or not to adjust "k", see comment on "BFS_kernel_multi_blk_inGPU" for more details 
		int *bfs_ptb_switch_k;
		int *bfs_ptb_num_t;//number of threads
		//whether to stay within a kernel, used in "BFS_kernel_multi_blk_inGPU"
		bool *bfs_ptb_stay;

		cudaMalloc((void**) &bfs_ptb_graph_edges, sizeof(Edge)*num_of_edges);
		cudaMalloc((void**) &bfs_ptb_graph_nodes, sizeof(Node)*num_of_nodes);

		cudaMalloc((void**) &bfs_ptb_color, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ptb_cost, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ptb_q1, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ptb_q2, sizeof(int)*num_of_nodes);
		cudaMalloc((void**) &bfs_ptb_tail, sizeof(int));
		cudaMalloc((void**) &bfs_ptb_front_cont, sizeof(int));

		cudaMalloc((void**) &bfs_ptb_max_nodes_per_block, sizeof(int));
		cudaMalloc((void**) &bfs_ptb_global_kt, sizeof(int));
		cudaMemcpy(bfs_ptb_global_kt,&zero, sizeof(int),cudaMemcpyHostToDevice);

		// //bind the texture memory with global memory
		cudaBindTexture(0, bfs_ptb_graph_node_ref, bfs_ptb_graph_nodes, sizeof(Node)*num_of_nodes);
		cudaBindTexture(0, bfs_ptb_graph_edge_ref, bfs_ptb_graph_edges, sizeof(Edge)*num_of_edges);

		cudaMalloc((void**) &bfs_ptb_switch_k, sizeof(int));
		cudaMalloc((void**) &bfs_ptb_num_t, sizeof(int));
		cudaMalloc( (void**) &bfs_ptb_stay, sizeof(bool));

		cudaMemcpy(bfs_ptb_tail, &h_top, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&bfs_ptb_cost[source], &zero, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&bfs_ptb_q1[0], &source, sizeof(int), cudaMemcpyHostToDevice);

		// int host_bfs_ptb_overflow = 0;
		int *bfs_ptb_overflow;
		cudaMalloc((void**) &bfs_ptb_overflow, sizeof(int));
	// ---------------------------------------------------------------------------------------

	// PRE running
    // ---------------------------------------------------------------------------------------
		int num_t;		//number of threads
		int k=0;		//BFS level index
		int num_of_blocks; 
		dim3 grid;
		dim3 threads;
		threads.x = 512;

		k = 0; num_t = 1; num_of_blocks = 1;
		grid.x = num_of_blocks;
		cudaMemcpy(bfs_ori_tail, &zero,sizeof(int),cudaMemcpyHostToDevice);
		BFS_in_GPU_kernel<<< grid, threads >>>(bfs_ori_q1, bfs_ori_q2, bfs_ori_graph_nodes, 
				bfs_ori_graph_edges, bfs_ori_color, bfs_ori_cost, num_t, bfs_ori_tail, GRAY0, k, ori_overflow);
		
		k = 1; num_t = 1000; num_of_blocks = 68;
		grid.x = num_of_blocks; 
		cudaMemcpy(bfs_ori_tail, &zero,sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(num_td, &num_t,sizeof(int), cudaMemcpyHostToDevice);
		BFS_kernel_multi_blk_inGPU <<< grid, threads >>>(bfs_ori_q2, bfs_ori_q1, bfs_ori_graph_nodes, 
			bfs_ori_graph_edges, bfs_ori_color, bfs_ori_cost, num_td, bfs_ori_tail, GRAY1, k,
			switch_kd, max_nodes_per_block_d, global_kt_d, ori_overflow);

		k = 1; num_t = 35969; num_of_blocks = 71;
		grid.x = num_of_blocks; 
		cudaMemcpy(bfs_ori_tail,&zero,sizeof(int),cudaMemcpyHostToDevice);
		ori_bfs<<< grid, threads >>>(bfs_ori_q2, bfs_ori_q1, bfs_ori_graph_nodes, 
					bfs_ori_graph_edges, bfs_ori_color, bfs_ori_cost, num_t, bfs_ori_tail, GRAY1, k, ori_overflow, 1);

		cudaMemcpy(bfs_ptb_q1, bfs_ori_q1, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToDevice);
		cudaMemcpy(bfs_ptb_q2, bfs_ori_q2, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToDevice);
		cudaMemcpy(bfs_ptb_graph_edges, bfs_ori_graph_edges, sizeof(Edge)*num_of_edges, cudaMemcpyDeviceToDevice);
		cudaMemcpy(bfs_ptb_graph_nodes, bfs_ori_graph_nodes, sizeof(Node)*num_of_nodes, cudaMemcpyDeviceToDevice);
		cudaMemcpy(bfs_ptb_color, bfs_ori_color, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToDevice);
		cudaMemcpy(bfs_ptb_cost, bfs_ori_cost, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToDevice);
		cudaMemcpy(bfs_ptb_overflow, ori_overflow, sizeof(int), cudaMemcpyDeviceToDevice);
	// ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
    dim3 wmma_grid;
    dim3 wmma_block;
	wmma_grid.x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	wmma_block.x = THREADS_PER_BLOCK;

	int wmma_grid_dim_x = (M_TILES * N_TILES) / (BLOCK_COL_TILES * BLOCK_ROW_TILES);
	int wmma_block_dim_x = wmma_block.x;
	wmma_grid.x = wmma_blks == 0 ? wmma_grid_dim_x : SM_NUM * wmma_blks;
	wmma_block.x = THREADS_PER_BLOCK;

	int SHMEM_SZ = WMMA_M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * WMMA_N * (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float);
	cudaErrCheck(cudaFuncSetAttribute(
		ptb_tzgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
	if (wmma_blks != 0) {
        SHMEM_SZ = 0;
    }

    printf("[PTB] Running with tzgemm...\n");
    printf("[PTB] wmma_grid -- %d * %d wmma_block -- %d * %d \n", wmma_grid.x, wmma_grid.y, wmma_block.x, wmma_block.y);
    cudaErrCheck(cudaEventRecord(startKERNEL));
    checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ptb_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
    cudaErrCheck(cudaEventRecord(stopKERNEL));
    cudaErrCheck(cudaEventSynchronize(stopKERNEL));
    cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    printf("[PTB] tzgemm took %f ms\n\n", kernel_time);
    serial_time += kernel_time;
	// printf("---init---\n");
	// float ori_fspmv_time = fspmv_call(streams[0], 0);
	// serial_time += ori_fspmv_time;
	// printf("\n");

	// SOLO running
    // ---------------------------------------------------------------------------------------
	k = 2; num_t = 215791; num_of_blocks = 422 * 512 / MAX_THREADS_PER_BLOCK;
	dim3 bfs_grid;
	dim3 bfs_block;
	bfs_block.x = MAX_THREADS_PER_BLOCK;
	bfs_grid.x = num_of_blocks; 

	printf("[ORI] Running with bfs...\n");
    printf("[ORI] bfs_grid -- %d * %d bfs_block -- %d * %d \n", bfs_grid.x, bfs_grid.y, bfs_block.x, bfs_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	cudaMemcpy(bfs_ori_tail, &zero, sizeof(int), cudaMemcpyHostToDevice);
	checkKernelErrors((ori_bfs<<< bfs_grid, bfs_block >>>(bfs_ori_q1, bfs_ori_q2, bfs_ori_graph_nodes, 
			bfs_ori_graph_edges, bfs_ori_color, bfs_ori_cost, num_t, bfs_ori_tail, GRAY0, k, ori_overflow, bfs_iter)));

	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[ORI] bfs took %f ms\n\n", kernel_time);

	// PTB running
    // ---------------------------------------------------------------------------------------
	int bfs_grid_dim_x = bfs_grid.x;
	int bfs_block_dim_x = bfs_block.x;
	k = 2; num_t = 215791; num_of_blocks = 422 * 512 / MAX_THREADS_PER_BLOCK;
    bfs_grid.x = bfs_blks == 0 ? bfs_grid_dim_x : SM_NUM * bfs_blks;
	// printf("[PTB] Running with bfs...\n");
    // printf("[PTB] bfs_grid -- %d * %d bfs_block -- %d * %d \n", bfs_grid.x, bfs_grid.y, bfs_block.x, bfs_block.y);
	cudaMemcpy(bfs_ptb_tail, &zero, sizeof(int), cudaMemcpyHostToDevice);

    serial_time += kernel_time;
	
	// MIX running 
    // ----------------------------------------------------------------------------------------------------------------------
	if (mixwarp == 1) {
		dim3 mix_grid, mix_block;
        mix_grid.x = (bfs_grid.x > wmma_grid.x) ? bfs_grid.x : wmma_grid.x;
        mix_grid.y = 1;
        mix_block.x = bfs_block.x + wmma_block.x;
        mix_block.y = 1;
    	printf("[PTB] bfs_grid -- %d * %d bfs_block -- %d * %d \n", bfs_grid.x, bfs_grid.y, bfs_block.x, bfs_block.y);
		printf("[MIX] mix_grid -- %d * %d mix_block -- %d * %d \n", mix_grid.x, mix_grid.y, mix_block.x, mix_block.y);

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((mix_kernel <<<mix_grid, mix_block>>> (
			// wmma parameters
			wmma_ori_a, wmma_ori_b, wmma_ori_c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, wmma_block_dim_x, wmma_iter,
			// bfs parameters
			bfs_ptb_q1, bfs_ptb_q2, bfs_ptb_graph_nodes, 
			bfs_ptb_graph_edges, bfs_ptb_color, bfs_ptb_cost, num_t, bfs_ptb_tail, GRAY0, k, bfs_ptb_overflow, 
			bfs_grid_dim_x, bfs_block_dim_x,
			bfs_iter
		)));
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[PETS] mix took %f ms\n\n", kernel_time);
	} else if (mixwarp == 2) {
		
		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		// fspmv_call(streams[0], 1);
		checkKernelErrors((ptb_bfs<<< bfs_grid, bfs_block, 0, streams[1] >>>(bfs_ptb_q1, bfs_ptb_q2, bfs_ptb_graph_nodes, 
			bfs_ptb_graph_edges, bfs_ptb_color, bfs_ptb_cost, num_t, bfs_ptb_tail, GRAY0, k, bfs_ptb_overflow, 
			bfs_grid_dim_x, bfs_block_dim_x,
			bfs_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    	printf("[PTB] bfs_grid -- %d * %d bfs_block -- %d * %d \n", bfs_grid.x, bfs_grid.y, bfs_block.x, bfs_block.y);
		printf("[STREAMP] mix took %f ms\n\n", kernel_time);
	} else {
		bfs_block.x = MAX_THREADS_PER_BLOCK;
		bfs_grid.x = num_of_blocks; 

		cudaErrCheck(cudaEventRecord(startKERNEL));
		checkKernelErrors((ptb_tzgemm<<<wmma_grid, wmma_block, SHMEM_SZ, streams[0]>>>(wmma_ori_a, wmma_ori_b, wmma_ori_c, 
							MATRIX_M, MATRIX_N, MATRIX_K,
							// alpha, beta,
							wmma_grid_dim_x, wmma_block_dim_x, wmma_iter)));
		checkKernelErrors((ori_bfs<<< bfs_grid, bfs_block, 0, streams[1] >>>(bfs_ptb_q1, bfs_ptb_q2, bfs_ptb_graph_nodes, 
			bfs_ptb_graph_edges, bfs_ptb_color, bfs_ptb_cost, num_t, bfs_ptb_tail, GRAY0, k, bfs_ptb_overflow, 
			bfs_iter)));
		
		cudaErrCheck(cudaEventRecord(stopKERNEL));
		cudaErrCheck(cudaEventSynchronize(stopKERNEL));
		cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
		printf("[STREAMO] mix took %f ms\n\n", kernel_time);
	}

	printf("[STAT] Overlap rate: %.2f\n", (serial_time - kernel_time) * 100 / serial_time);
    printf("[STAT] Throughput speedup: %.2f\n", (serial_time / kernel_time - 1) * 100);

	// Checking results
    // ---------------------------------------------------------------------------------------
    printf("Checking results...\n");
    cudaErrCheck(cudaMemcpy(host_wmma_ori_c, wmma_ori_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(host_wmma_ptb_c, wmma_ptb_c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaMemcpy(host_bfs_ori_cost, bfs_ori_cost, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_bfs_bfs_ori_color, bfs_ori_color, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_bfs_ptb_cost, bfs_ptb_cost, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_bfs_bfs_ptb_color, bfs_ptb_color, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);

    int errors = 0;
    for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
        float v1 = host_wmma_ori_c[i];
        float v2 = host_wmma_ptb_c[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
		if (i < 3) printf("%d %f %f\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("[WMMA] ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("[WMMA] Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }
    errors = 0;
    for (int i = 0; i < num_of_nodes; i++) {
        int v1 = host_bfs_ori_cost[i];
        int v2 = host_bfs_ptb_cost[i];
        if (v1 - v2 != 0) {
            errors++;
            if (errors < 5) printf("%d %d\n", v1, v2);
        }
		if (i < 3) printf("%d %d %d\n", i, v1, v2);
    }
    if (errors > 0) {
        printf("ORIGIN VERSION does not agree with MY VERSION! %d errors!\n", errors);
    }
    else {
        printf("Results verified: ORIGIN VERSION and MY VERSION agree.\n");
    }

    cudaErrCheck(cudaEventDestroy(startKERNEL));
    cudaErrCheck(cudaEventDestroy(stopKERNEL));

    cudaErrCheck(cudaDeviceReset());
    return 0;
}