__global__ void ori_bfs(int *q1, 
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
		   int iteration)  
{	
	for (int loop = 0; loop < iteration; loop++) {
		__shared__ LocalQueues local_q;
		__shared__ int prefix_q[NUM_BIN];//the number of elementss in the w-queues ahead of
		//current w-queue, a.k.a prefix sum
		__shared__ int shift;

		if(threadIdx.x < NUM_BIN) {
			local_q.reset(threadIdx.x, blockDim);
		}
		__syncthreads();

		//first, propagate and add the new frontier elements into w-queues
		int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
		if( tid < no_of_nodes) {
			// Visit a node from the current frontier; update costs, colors, and
			// output queue
			visit_node(q1[tid], threadIdx.x & MOD_OP, local_q, overflow,
					g_color, g_cost, gray_shade);
		}
		__syncthreads();

		// Compute size of the output and allocate space in the global queue
		if(threadIdx.x == 0) {
			//now calculate the prefix sum
			int tot_sum = local_q.size_prefix_sum(prefix_q);
			//the offset or "shift" of the block-level queue within the
			//grid-level queue is determined by atomic operation
			shift = atomicAdd(tail,tot_sum);
		}
		__syncthreads();

		//now copy the elements from w-queues into grid-level queues.
		//Note that we have bypassed the copy to/from block-level queues for efficiency reason
		local_q.concatenate(q2 + shift, prefix_q);
	}
}


__global__ void ptb_bfs(int *q1, 
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
           int iteration
           ) 
{	
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }
        int block_id_x = block_pos;

        for (int loop = 0; loop < iteration; loop++) {
            __shared__ LocalQueues local_q;
            __shared__ int prefix_q[NUM_BIN];//the number of elementss in the w-queues ahead of
            //current w-queue, a.k.a prefix sum
            __shared__ int shift;

            if(thread_id_x < NUM_BIN) {
                local_q.reset(thread_id_x, block_dimension_x);
            }
            __syncthreads();

            //first, propagate and add the new frontier elements into w-queues
            int tid = block_id_x * MAX_THREADS_PER_BLOCK + thread_id_x;
            if( tid < no_of_nodes) {
                // Visit a node from the current frontier; update costs, colors, and
                // output queue
                ptb_visit_node(q1[tid], thread_id_x & MOD_OP, local_q, overflow,
                        g_color, g_cost, gray_shade);
            }
            __syncthreads();

            // Compute size of the output and allocate space in the global queue
            if(thread_id_x == 0){
                //now calculate the prefix sum
                int tot_sum = local_q.size_prefix_sum(prefix_q);
                //the offset or "shift" of the block-level queue within the
                //grid-level queue is determined by atomic operation
                shift = atomicAdd(tail,tot_sum);
            }
            __syncthreads();

            //now copy the elements from w-queues into grid-level queues.
            //Note that we have bypassed the copy to/from block-level queues for efficiency reason
            local_q.concatenate(q2 + shift, prefix_q);
        }
    }
}


__device__ void mix_bfs(int *q1, 
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
		   int thread_step,
           int iteration
           ) 
{
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x - thread_step;

    for (;; block_pos += BFS_GRID_DIM) {
        if (block_pos >= grid_dimension_x) {
            return;
        }
        int block_id_x = block_pos;

        for (int loop = 0; loop < iteration; loop++) {
            __shared__ LocalQueues local_q;
            __shared__ int prefix_q[NUM_BIN];//the number of elementss in the w-queues ahead of
            //current w-queue, a.k.a prefix sum
            __shared__ int shift;

            if(thread_id_x < NUM_BIN) {
                local_q.reset(thread_id_x, block_dimension_x);
            }
            // __syncthreads();
        	asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(block_dimension_x) : "memory");

            //first, propagate and add the new frontier elements into w-queues
            int tid = block_id_x * MAX_THREADS_PER_BLOCK + thread_id_x;
            if( tid < no_of_nodes) {
                // Visit a node from the current frontier; update costs, colors, and
                // output queue
                visit_node(q1[tid], thread_id_x & MOD_OP, local_q, overflow,
                        g_color, g_cost, gray_shade);
            }
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(block_dimension_x) : "memory");

            // Compute size of the output and allocate space in the global queue
            if(thread_id_x == 0){
                //now calculate the prefix sum
                int tot_sum = local_q.size_prefix_sum(prefix_q);
                //the offset or "shift" of the block-level queue within the
                //grid-level queue is determined by atomic operation
                shift = atomicAdd(tail,tot_sum);
            }
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(block_dimension_x) : "memory");

            //now copy the elements from w-queues into grid-level queues.
            //Note that we have bypassed the copy to/from block-level queues for efficiency reason
            local_q.concatenate(q2 + shift, prefix_q);
        }
    }
}

__device__ void mix_bfs_float(int *q1, 
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
		   int thread_step,
           int iteration
           ) 
{
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x - thread_step;

    for (;; block_pos += BFS_GRID_DIM) {
        if (block_pos >= grid_dimension_x) {
            return;
        }
        int block_id_x = block_pos;

        for (int loop = 0; loop < iteration; loop++) {
            __shared__ LocalQueues local_q;
            __shared__ int prefix_q[NUM_BIN];//the number of elementss in the w-queues ahead of
            //current w-queue, a.k.a prefix sum
            __shared__ int shift;

            if(thread_id_x < NUM_BIN) {
                local_q.reset(thread_id_x, block_dimension_x);
            }
            // __syncthreads();
        	asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(block_dimension_x) : "memory");

            //first, propagate and add the new frontier elements into w-queues
            int tid = block_id_x * MAX_THREADS_PER_BLOCK + thread_id_x;
            if( tid < no_of_nodes) {
                // Visit a node from the current frontier; update costs, colors, and
                // output queue
                visit_node(q1[tid], thread_id_x & MOD_OP, local_q, overflow,
                        g_color, g_cost, gray_shade);
            }
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(block_dimension_x) : "memory");

            // Compute size of the output and allocate space in the global queue
            if(thread_id_x == 0){
                //now calculate the prefix sum
                int tot_sum = local_q.size_prefix_sum(prefix_q);
                //the offset or "shift" of the block-level queue within the
                //grid-level queue is determined by atomic operation
                shift = atomicAdd(tail,tot_sum);
            }
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(0), "r"(block_dimension_x) : "memory");

            //now copy the elements from w-queues into grid-level queues.
            //Note that we have bypassed the copy to/from block-level queues for efficiency reason
            local_q.concatenate(q2 + shift, prefix_q);
        }
    }
}


__global__ void BFS_in_GPU_kernel(int *q1, 
                  int *q2, 
                  Node *g_graph_nodes, 
                  Edge *g_graph_edges, 
                  int *g_color, 
                  int *g_cost, 
                  int no_of_nodes, 
                  int *tail, 
                  int gray_shade, 
                  int k,
                  int *overflow) 
{
    __shared__ LocalQueues local_q;
    __shared__ int prefix_q[NUM_BIN];

    //next/new wave front
    __shared__ int next_wf[MAX_THREADS_PER_BLOCK];
    __shared__ int  tot_sum;
    if(threadIdx.x == 0) {
        tot_sum = 0;//total number of new frontier nodes
    }
    
    while(1){//propage through multiple BFS levels until the wavfront overgrows one-block limit
        if(threadIdx.x < NUM_BIN){
            local_q.reset(threadIdx.x, blockDim);
        }
        __syncthreads();
        int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
    
        if( tid<no_of_nodes) {
            int pid;
            if(tot_sum == 0)//this is the first BFS level of current kernel call
                pid = q1[tid];  
            else
                pid = next_wf[tid];//read the current frontier info from last level's propagation

            // Visit a node from the current frontier; update costs, colors, and
            // output queue
            visit_node(pid, threadIdx.x & MOD_OP, local_q, overflow,
                g_color, g_cost, gray_shade);
        }
        __syncthreads();
        if(threadIdx.x == 0){
            *tail = tot_sum = local_q.size_prefix_sum(prefix_q);
        }
        __syncthreads();

        if(tot_sum == 0)//the new frontier becomes empty; BFS is over
            return;
        
        if(tot_sum <= MAX_THREADS_PER_BLOCK) {
            //the new frontier is still within one-block limit;
            //stay in current kernel
            local_q.concatenate(next_wf, prefix_q);
            __syncthreads();
            no_of_nodes = tot_sum;
            if(threadIdx.x == 0) {
                if(gray_shade == GRAY0)
                    gray_shade = GRAY1;
                else
                    gray_shade = GRAY0;
            }
        } else {
            //the new frontier outgrows one-block limit; terminate current kernel
            local_q.concatenate(q2, prefix_q);
            return;
        }
    }//while
}


__global__ void BFS_kernel_multi_blk_inGPU(int *q1,
                           int *q2, 
                           Node *g_graph_nodes, 
                           Edge *g_graph_edges, 
                           int *g_color, 
                           int *g_cost, 
                           int *no_of_nodes, 
                           int *tail, 
                           int gray_shade, 
                           int k,   
                           int *switch_k, 
                           int *max_nodes_per_block, 
                           int *global_kt,
                           int *overflow) 
{
    __shared__ LocalQueues local_q;
    __shared__ int prefix_q[NUM_BIN];
    __shared__ int shift;
    __shared__ int no_of_nodes_sm;
    __shared__ int odd_time;// the odd level of propagation within current kernel
    if(threadIdx.x == 0){
        odd_time = 1;//true;
        if(blockIdx.x == 0)
            no_of_nodes_vol = *no_of_nodes;
    }

    int kt = atomicOr(global_kt,0);// the total count of GPU global synchronization 
    while (1){//propagate through multiple levels
        if(threadIdx.x < NUM_BIN){
            local_q.reset(threadIdx.x, blockDim);
        }
        if(threadIdx.x == 0)
            no_of_nodes_sm = no_of_nodes_vol; 
        __syncthreads();

        int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
        if( tid<no_of_nodes_sm) {
            // Read a node ID from the current input queue
            int *input_queue = odd_time ? q1 : q2;
            int pid = atomicOr((int *)&input_queue[tid], 0);

            // Visit a node from the current frontier; update costs, colors, and
            // output queue
            visit_node(pid, threadIdx.x & MOD_OP, local_q, overflow,
                g_color, g_cost, gray_shade);
        }
        __syncthreads();

        // Compute size of the output and allocate space in the global queue
        if(threadIdx.x == 0){
            int tot_sum = local_q.size_prefix_sum(prefix_q);
            shift = atomicAdd(tail, tot_sum);
        }
        __syncthreads();

        // Copy to the current output queue in global memory
        int *output_queue = odd_time ? q2 : q1;
        local_q.concatenate(output_queue + shift, prefix_q);

        if(threadIdx.x == 0) {
            odd_time = (odd_time+1)%2;
            if(gray_shade == GRAY0)
                gray_shade = GRAY1;
            else
                gray_shade = GRAY0;
        }

        //synchronize among all the blks
        start_global_barrier(kt+1);
        if(blockIdx.x == 0 && threadIdx.x == 0) {
            stay_vol = 0;
            if(*tail< NUM_SM*MAX_THREADS_PER_BLOCK && *tail > MAX_THREADS_PER_BLOCK) {
                stay_vol = 1;
                no_of_nodes_vol = *tail;
                *tail = 0;
            }
        }
        start_global_barrier(kt+2);
        kt+= 2;
        if(stay_vol == 0) {
            if(blockIdx.x == 0 && threadIdx.x == 0) {
                *global_kt = kt;
                *switch_k = (odd_time+1)%2;
                *no_of_nodes = no_of_nodes_vol;
            }
            return;
        }
    }
}
