#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <deque>
#include <iostream>

#include <curand.h>
#include <cublas_v2.h>

#define ITERATION 1

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


#include "header/bfs_header.h"
#include "file_t/bfs_kernel.cu"


int main(int argc, char** argv) 
{
	int bfs_blks = 1;
	int bfs_iter = 1;
	// int bfs_iter = 8500;
    if (argc == 3) {
        bfs_blks = atoi(argv[1]);
        bfs_iter = atoi(argv[2]);
    }

	// variables
    // ---------------------------------------------------------------------------------------
    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    cudaErrCheck(cudaEventCreate(&startKERNEL));
    cudaErrCheck(cudaEventCreate(&stopKERNEL));

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
		char file_dat[] = "file_t/bfs_input.dat";
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
	k = 2; num_t = 215791; num_of_blocks = 422 * 512 / MAX_THREADS_PER_BLOCK;
	dim3 bfs_grid;
	dim3 bfs_block;
	bfs_block.x = MAX_THREADS_PER_BLOCK;
	bfs_grid.x = num_of_blocks; 

	printf("[ORI] Running with bfs...\n");
    printf("[ORI] bfs_grid -- %d * %d bfs_block -- %d * %d \n", bfs_grid.x, bfs_grid.y, bfs_block.x, bfs_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	cudaMemcpy(bfs_ori_tail, &zero, sizeof(int), cudaMemcpyHostToDevice);
	for (int i = 0; i < ITERATION; i++) {
		checkKernelErrors((ori_bfs<<< bfs_grid, bfs_block >>>(bfs_ori_q1, bfs_ori_q2, 
				bfs_ori_graph_nodes, bfs_ori_graph_edges, bfs_ori_color, bfs_ori_cost, 
				num_t, bfs_ori_tail, GRAY0, k, ori_overflow, bfs_iter)));
	}
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
	printf("[PTB] Running with bfs...\n");
    printf("[PTB] bfs_grid -- %d * %d bfs_block -- %d * %d \n", bfs_grid.x, bfs_grid.y, bfs_block.x, bfs_block.y);

	cudaErrCheck(cudaEventRecord(startKERNEL));
	cudaMemcpy(bfs_ptb_tail, &zero, sizeof(int), cudaMemcpyHostToDevice);
	for (int i = 0; i < ITERATION; i++) {
		checkKernelErrors((ptb_bfs<<< bfs_grid, bfs_block >>>(bfs_ptb_q1, bfs_ptb_q2, 
				bfs_ptb_graph_nodes, bfs_ptb_graph_edges, bfs_ptb_color, bfs_ptb_cost, num_t, bfs_ptb_tail, GRAY0, k, bfs_ptb_overflow,
				bfs_grid_dim_x, bfs_block_dim_x, bfs_iter)));
	}
	cudaErrCheck(cudaEventRecord(stopKERNEL));
	cudaErrCheck(cudaEventSynchronize(stopKERNEL));
	cudaErrCheck(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
	printf("[PTB] bfs took %f ms\n\n", kernel_time);

	// Checking results
    // ---------------------------------------------------------------------------------------
	cudaMemcpy(host_bfs_ori_cost, bfs_ori_cost, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_bfs_bfs_ori_color, bfs_ori_color, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_bfs_ptb_cost, bfs_ptb_cost, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_bfs_bfs_ptb_color, bfs_ptb_color, sizeof(int)*num_of_nodes, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(bfs_ori_graph_node_ref);
	cudaUnbindTexture(bfs_ori_graph_edge_ref);
	cudaUnbindTexture(bfs_ptb_graph_node_ref);
	cudaUnbindTexture(bfs_ptb_graph_edge_ref);

	printf("Checking results...\n");
	int errors = 0;
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
	
	return 0;
}
