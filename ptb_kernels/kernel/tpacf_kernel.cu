
__global__ void ori_tpacf( hist_t* histograms, float* all_x_data, float* all_y_data, 
    float* all_z_data, int NUM_SETS, int NUM_ELEMENTS, int iteration ) {
    unsigned int bx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    bool do_self = (bx < (NUM_SETS + 1));

    float* data_x;
    float* data_y;
    float* data_z;
    float* random_x;
    float* random_y;
    float* random_z;

    __shared__ struct cartesian data_s[BLOCK_SIZE];
    // struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS]; // 640B <1k  

        for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE ) {
            if(w+tid < NUM_BINS*NUM_HISTOGRAMS) {
                warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
            }
        }

        // Get stuff into shared memory to kick off the loop.
        if( !do_self) {
            data_x = all_x_data;
            data_y = all_y_data;
            data_z = all_z_data;
            random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        } else {
            random_x = all_x_data + NUM_ELEMENTS * (bx);
            random_y = all_y_data + NUM_ELEMENTS * (bx);
            random_z = all_z_data + NUM_ELEMENTS * (bx);
            
            data_x = random_x;
            data_y = random_y;
            data_z = random_z;
        }

        // Iterate over all data points
        for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE ) {
            // load current set of data into shared memory
            // (total of BLOCK_SIZE points loaded)
            if( tid + i < NUM_ELEMENTS ) { // reading outside of bounds is a-okay
                data_s[tid] = (struct cartesian){data_x[tid + i], data_y[tid + i], data_z[tid + i]};
            }
        
            __syncthreads();

            // Iterate over all random points
            for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; 
                    j += BLOCK_SIZE) {
                // load current random point values
                float random_x_s;
                float random_y_s;
                float random_z_s;
            
                if(tid + j < NUM_ELEMENTS) {
                    random_x_s = random_x[tid + j];
                    random_y_s = random_y[tid + j];
                    random_z_s = random_z[tid + j];
                }

                // Iterate for all elements of current set of data points 
                // (BLOCK_SIZE iterations per thread)
                // Each thread calcs against 1 random point within cur set of random
                // (so BLOCK_SIZE threads covers all random points within cur set)
                for(unsigned int k = 0; 
                    (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                    k += 1) {
                    // do actual calculations on the values:
                    float distance = 
                        data_s[k].x * random_x_s +
                        data_s[k].y * random_y_s +
                        data_s[k].z * random_z_s;

                    unsigned int bin_index;

                    // run binary search to find bin_index
                    unsigned int min = 0;
                    unsigned int max = NUM_BINS;
                    {
                        unsigned int k2;
                        
                        while (max > min+1) {
                            k2 = (min + max) / 2;
                            if (distance >= dev_binb[k2]) {
                                max = k2;
                            } else {
                                min = k2;
                            } 
                        }
                        bin_index = max - 1;
                    }

                    unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                    if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                    (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS)) {
                        atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                    }
                }
            }

            __syncthreads();
        }

        // coalesce the histograms in a block
        unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; offset >>= 1)
        {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
            bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
            {
                __syncthreads();
                if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                {
                    unsigned long sum =
                    warp_hists[bin_base + bin_index][warp_index] + 
                    warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        __syncthreads();

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = histograms + NUM_BINS * bx;
        if(tid < NUM_BINS)
        {
            hist_base[tid] = warp_hists[tid][0];
        }

        __syncthreads();
}


__global__ void ptb_tpacf( hist_t* histograms, float* all_x_data, float* all_y_data, 
    float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    // int thread_id_y = threadIdx.x / block_dimension_x;

    __shared__ struct cartesian data_s[BLOCK_SIZE];
    // struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        // int block_id_y = block_pos / grid_dimension_x;

        bool do_self = (block_id_x < (NUM_SETS + 1));
        int tid = thread_id_x;
        int bx = block_id_x;
        float* data_x;
        float* data_y;
        float* data_z;
        float* random_x;
        float* random_y;
        float* random_z;

            for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE ) {
                if(w+tid < NUM_BINS*NUM_HISTOGRAMS) {
                    warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
                }
            }

            // Get stuff into shared memory to kick off the loop.
            if( !do_self) {
                data_x = all_x_data;
                data_y = all_y_data;
                data_z = all_z_data;
                random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
                random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
                random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
            } else {
                random_x = all_x_data + NUM_ELEMENTS * (bx);
                random_y = all_y_data + NUM_ELEMENTS * (bx);
                random_z = all_z_data + NUM_ELEMENTS * (bx);
                
                data_x = random_x;
                data_y = random_y;
                data_z = random_z;
            }

            // Iterate over all data points
            for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE ) {
                // load current set of data into shared memory
                // (total of BLOCK_SIZE points loaded)
                if( tid + i < NUM_ELEMENTS ) { // reading outside of bounds is a-okay
                    data_s[tid] = (struct cartesian){data_x[tid + i], data_y[tid + i], data_z[tid + i]};
                }
            
                __syncthreads();

                // Iterate over all random points
                for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; 
                        j += BLOCK_SIZE) {
                    // load current random point values
                    float random_x_s;
                    float random_y_s;
                    float random_z_s;
                
                    if(tid + j < NUM_ELEMENTS) {
                        random_x_s = random_x[tid + j];
                        random_y_s = random_y[tid + j];
                        random_z_s = random_z[tid + j];
                    }

                    // Iterate for all elements of current set of data points 
                    // (BLOCK_SIZE iterations per thread)
                    // Each thread calcs against 1 random point within cur set of random
                    // (so BLOCK_SIZE threads covers all random points within cur set)
                    for(unsigned int k = 0; 
                        (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                        k += 1) {
                        // do actual calculations on the values:
                        float distance = 
                            data_s[k].x * random_x_s +
                            data_s[k].y * random_y_s +
                            data_s[k].z * random_z_s;

                        unsigned int bin_index;

                        // run binary search to find bin_index
                        unsigned int min = 0;
                        unsigned int max = NUM_BINS;
                        {
                            unsigned int k2;
                            
                            while (max > min+1) {
                                k2 = (min + max) / 2;
                                if (distance >= dev_binb[k2]) {
                                    max = k2;
                                } else { 
                                    min = k2;
                                }
                            }
                            bin_index = max - 1;
                        }

                        unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                        if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                            (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS)) {
                            atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                        }
                    }
                }

                __syncthreads();
            }

            // coalesce the histograms in a block
            unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
            unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
            for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; offset >>= 1) {
                for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
                bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1)) {
                    __syncthreads();
                    if(warp_index < offset && bin_base+bin_index < NUM_BINS ) {
                        unsigned long sum =
                        warp_hists[bin_base + bin_index][warp_index] + 
                        warp_hists[bin_base + bin_index][warp_index+offset];
                        warp_hists[bin_base + bin_index][warp_index] = sum;
                    }
                }
            }

            __syncthreads();

            // Put the results back in the real histogram
            // warp_hists[x][0] holds sum of all locations of bin x
            hist_t* hist_base = histograms + NUM_BINS * bx;
            if(tid < NUM_BINS) {
                hist_base[tid] = warp_hists[tid][0];
            }

            __syncthreads();
        }
}


__device__ void mix_tpacf( hist_t* histograms, float* all_x_data, float* all_y_data, 
    float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
    int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, 
    int thread_step, int iteration) {

    unsigned int block_pos = blockIdx.x;
    int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    // __shared__ struct cartesian data_s[BLOCK_SIZE];
    struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += TPACF_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        // int block_id_y = block_pos / grid_dimension_x;


            unsigned int bx = block_id_x;
            unsigned int tid = thread_id_x;
            bool do_self = (bx < (NUM_SETS + 1));

            float* data_x;
            float* data_y;
            float* data_z;
            float* random_x;
            float* random_y;
            float* random_z;

            for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
            {
                if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
                {
                    warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
                }
            }

            // Get stuff into shared memory to kick off the loop.
            if( !do_self)
            {
                data_x = all_x_data;
                data_y = all_y_data;
                data_z = all_z_data;
                random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
                random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
                random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
            }
            else
            {
                random_x = all_x_data + NUM_ELEMENTS * (bx);
                random_y = all_y_data + NUM_ELEMENTS * (bx);
                random_z = all_z_data + NUM_ELEMENTS * (bx);
                
                data_x = random_x;
                data_y = random_y;
                data_z = random_z;
            }

            // Iterate over all data points
            for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
            {
                // load current set of data into shared memory
                // (total of BLOCK_SIZE points loaded)
                if( tid + i < NUM_ELEMENTS )
                { // reading outside of bounds is a-okay
                    data_s[tid] = (struct cartesian)
                            {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
                }
                
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                // Iterate over all random points
                for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; j += BLOCK_SIZE)
                {
                    // load current random point values
                    float random_x_s;
                    float random_y_s;
                    float random_z_s;
                    
                    if(tid + j < NUM_ELEMENTS)
                    {
                        random_x_s = random_x[tid + j];
                        random_y_s = random_y[tid + j];
                        random_z_s = random_z[tid + j];
                    }

                    // Iterate for all elements of current set of data points 
                    // (BLOCK_SIZE iterations per thread)
                    // Each thread calcs against 1 random point within cur set of random
                    // (so BLOCK_SIZE threads covers all random points within cur set)
                    for(unsigned int k = 0; 
                        (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                        k += 1)
                    {
                        // do actual calculations on the values:
                        float distance = 
                            data_s[k].x * random_x_s +
                            data_s[k].y * random_y_s +
                            data_s[k].z * random_z_s;

                        unsigned int bin_index;

                        // run binary search to find bin_index
                        unsigned int min = 0;
                        unsigned int max = NUM_BINS;
                        {
                            unsigned int k2;
                            
                            while (max > min+1)
                            {
                                k2 = (min + max) / 2;
                                if (distance >= dev_binb[k2]) 
                                max = k2;
                                else 
                                min = k2;
                            }
                            bin_index = max - 1;
                        }

                        unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                        if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                        (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
                        {
                            atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                        }
                    }
                }
            }

            // coalesce the histograms in a block
            unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
            unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
            for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
            offset >>= 1)
            {
                for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
                bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
                {
                    // __syncthreads();
                    asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                    if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                    {
                        unsigned long sum =
                        warp_hists[bin_base + bin_index][warp_index] + 
                        warp_hists[bin_base + bin_index][warp_index+offset];
                        warp_hists[bin_base + bin_index][warp_index] = sum;
                    }
                }
            }

            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

            // Put the results back in the real histogram
            // warp_hists[x][0] holds sum of all locations of bin x
            hist_t* hist_base = histograms + NUM_BINS * bx;
            if(tid < NUM_BINS)
            {
                hist_base[tid] = warp_hists[tid][0];
            }
        }
}


__device__ void general_ptb_tpacf0(hist_t* histograms, float* all_x_data, float* all_y_data, float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;


    // __shared__ struct cartesian data_s[BLOCK_SIZE];
    // TODO : in mix version, why no shared? 
    __shared__ struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        unsigned int bx = block_id_x;
        unsigned int tid = thread_id_x;
        bool do_self = (bx < (NUM_SETS + 1));

        float* data_x;
        float* data_y;
        float* data_z;
        float* random_x;
        float* random_y;
        float* random_z;

        for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
        {
            if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
            {
                warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
            }
        }

        // Get stuff into shared memory to kick off the loop.
        if( !do_self)
        {
            data_x = all_x_data;
            data_y = all_y_data;
            data_z = all_z_data;
            random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        }
        else
        {
            random_x = all_x_data + NUM_ELEMENTS * (bx);
            random_y = all_y_data + NUM_ELEMENTS * (bx);
            random_z = all_z_data + NUM_ELEMENTS * (bx);
            
            data_x = random_x;
            data_y = random_y;
            data_z = random_z;
        }

        // Iterate over all data points
        for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
        {
            // load current set of data into shared memory
            // (total of BLOCK_SIZE points loaded)
            if( tid + i < NUM_ELEMENTS )
            { // reading outside of bounds is a-okay
                data_s[tid] = (struct cartesian)
                        {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
            }
            
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

            // Iterate over all random points
            for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; j += BLOCK_SIZE)
            {
                // load current random point values
                float random_x_s;
                float random_y_s;
                float random_z_s;
                
                if(tid + j < NUM_ELEMENTS)
                {
                    random_x_s = random_x[tid + j];
                    random_y_s = random_y[tid + j];
                    random_z_s = random_z[tid + j];
                }

                // Iterate for all elements of current set of data points 
                // (BLOCK_SIZE iterations per thread)
                // Each thread calcs against 1 random point within cur set of random
                // (so BLOCK_SIZE threads covers all random points within cur set)
                for(unsigned int k = 0; 
                    (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                    k += 1)
                {
                    // do actual calculations on the values:
                    float distance = 
                        data_s[k].x * random_x_s +
                        data_s[k].y * random_y_s +
                        data_s[k].z * random_z_s;

                    unsigned int bin_index;

                    // run binary search to find bin_index
                    unsigned int min = 0;
                    unsigned int max = NUM_BINS;
                    {
                        unsigned int k2;
                        
                        while (max > min+1)
                        {
                            k2 = (min + max) / 2;
                            if (distance >= dev_binb[k2]) 
                            max = k2;
                            else 
                            min = k2;
                        }
                        bin_index = max - 1;
                    }

                    unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                    if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                    (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
                    {
                        atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                    }
                }
            }
            // TODO: why no sync?
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
        }

        // coalesce the histograms in a block
        unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
        offset >>= 1)
        {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
            bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
            {
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                {
                    unsigned long sum =
                    warp_hists[bin_base + bin_index][warp_index] + 
                    warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        // __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = histograms + NUM_BINS * bx;
        if(tid < NUM_BINS)
        {
            hist_base[tid] = warp_hists[tid][0];
        }
        // TODO: __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
    }
}

__device__ void general_ptb_tpacf0(hist_t* histograms, float* all_x_data, float* all_y_data, float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;


    // __shared__ struct cartesian data_s[BLOCK_SIZE];
    // TODO : in mix version, why no shared? 
    __shared__ struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        unsigned int bx = block_id_x;
        unsigned int tid = thread_id_x;
        bool do_self = (bx < (NUM_SETS + 1));

        float* data_x;
        float* data_y;
        float* data_z;
        float* random_x;
        float* random_y;
        float* random_z;

        for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
        {
            if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
            {
                warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
            }
        }

        // Get stuff into shared memory to kick off the loop.
        if( !do_self)
        {
            data_x = all_x_data;
            data_y = all_y_data;
            data_z = all_z_data;
            random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        }
        else
        {
            random_x = all_x_data + NUM_ELEMENTS * (bx);
            random_y = all_y_data + NUM_ELEMENTS * (bx);
            random_z = all_z_data + NUM_ELEMENTS * (bx);
            
            data_x = random_x;
            data_y = random_y;
            data_z = random_z;
        }

        // Iterate over all data points
        for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
        {
            // load current set of data into shared memory
            // (total of BLOCK_SIZE points loaded)
            if( tid + i < NUM_ELEMENTS )
            { // reading outside of bounds is a-okay
                data_s[tid] = (struct cartesian)
                        {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
            }
            
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

            // Iterate over all random points
            for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; j += BLOCK_SIZE)
            {
                // load current random point values
                float random_x_s;
                float random_y_s;
                float random_z_s;
                
                if(tid + j < NUM_ELEMENTS)
                {
                    random_x_s = random_x[tid + j];
                    random_y_s = random_y[tid + j];
                    random_z_s = random_z[tid + j];
                }

                // Iterate for all elements of current set of data points 
                // (BLOCK_SIZE iterations per thread)
                // Each thread calcs against 1 random point within cur set of random
                // (so BLOCK_SIZE threads covers all random points within cur set)
                for(unsigned int k = 0; 
                    (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                    k += 1)
                {
                    // do actual calculations on the values:
                    float distance = 
                        data_s[k].x * random_x_s +
                        data_s[k].y * random_y_s +
                        data_s[k].z * random_z_s;

                    unsigned int bin_index;

                    // run binary search to find bin_index
                    unsigned int min = 0;
                    unsigned int max = NUM_BINS;
                    {
                        unsigned int k2;
                        
                        while (max > min+1)
                        {
                            k2 = (min + max) / 2;
                            if (distance >= dev_binb[k2]) 
                            max = k2;
                            else 
                            min = k2;
                        }
                        bin_index = max - 1;
                    }

                    unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                    if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                    (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
                    {
                        atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                    }
                }
            }
            // TODO: why no sync?
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
        }

        // coalesce the histograms in a block
        unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
        offset >>= 1)
        {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
            bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
            {
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                {
                    unsigned long sum =
                    warp_hists[bin_base + bin_index][warp_index] + 
                    warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        // __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = histograms + NUM_BINS * bx;
        if(tid < NUM_BINS)
        {
            hist_base[tid] = warp_hists[tid][0];
        }
        // TODO: __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
    }
}

__device__ void general_ptb_tpacf0(hist_t* histograms, float* all_x_data, float* all_y_data, float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;


    // __shared__ struct cartesian data_s[BLOCK_SIZE];
    // TODO : in mix version, why no shared? 
    __shared__ struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        unsigned int bx = block_id_x;
        unsigned int tid = thread_id_x;
        bool do_self = (bx < (NUM_SETS + 1));

        float* data_x;
        float* data_y;
        float* data_z;
        float* random_x;
        float* random_y;
        float* random_z;

        for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
        {
            if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
            {
                warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
            }
        }

        // Get stuff into shared memory to kick off the loop.
        if( !do_self)
        {
            data_x = all_x_data;
            data_y = all_y_data;
            data_z = all_z_data;
            random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        }
        else
        {
            random_x = all_x_data + NUM_ELEMENTS * (bx);
            random_y = all_y_data + NUM_ELEMENTS * (bx);
            random_z = all_z_data + NUM_ELEMENTS * (bx);
            
            data_x = random_x;
            data_y = random_y;
            data_z = random_z;
        }

        // Iterate over all data points
        for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
        {
            // load current set of data into shared memory
            // (total of BLOCK_SIZE points loaded)
            if( tid + i < NUM_ELEMENTS )
            { // reading outside of bounds is a-okay
                data_s[tid] = (struct cartesian)
                        {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
            }
            
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

            // Iterate over all random points
            for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; j += BLOCK_SIZE)
            {
                // load current random point values
                float random_x_s;
                float random_y_s;
                float random_z_s;
                
                if(tid + j < NUM_ELEMENTS)
                {
                    random_x_s = random_x[tid + j];
                    random_y_s = random_y[tid + j];
                    random_z_s = random_z[tid + j];
                }

                // Iterate for all elements of current set of data points 
                // (BLOCK_SIZE iterations per thread)
                // Each thread calcs against 1 random point within cur set of random
                // (so BLOCK_SIZE threads covers all random points within cur set)
                for(unsigned int k = 0; 
                    (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                    k += 1)
                {
                    // do actual calculations on the values:
                    float distance = 
                        data_s[k].x * random_x_s +
                        data_s[k].y * random_y_s +
                        data_s[k].z * random_z_s;

                    unsigned int bin_index;

                    // run binary search to find bin_index
                    unsigned int min = 0;
                    unsigned int max = NUM_BINS;
                    {
                        unsigned int k2;
                        
                        while (max > min+1)
                        {
                            k2 = (min + max) / 2;
                            if (distance >= dev_binb[k2]) 
                            max = k2;
                            else 
                            min = k2;
                        }
                        bin_index = max - 1;
                    }

                    unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                    if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                    (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
                    {
                        atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                    }
                }
            }
            // TODO: why no sync?
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
        }

        // coalesce the histograms in a block
        unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
        offset >>= 1)
        {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
            bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
            {
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                {
                    unsigned long sum =
                    warp_hists[bin_base + bin_index][warp_index] + 
                    warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        // __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = histograms + NUM_BINS * bx;
        if(tid < NUM_BINS)
        {
            hist_base[tid] = warp_hists[tid][0];
        }
        // TODO: __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
    }
}

__device__ void general_ptb_tpacf0(hist_t* histograms, float* all_x_data, float* all_y_data, float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;


    // __shared__ struct cartesian data_s[BLOCK_SIZE];
    // TODO : in mix version, why no shared? 
    __shared__ struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        unsigned int bx = block_id_x;
        unsigned int tid = thread_id_x;
        bool do_self = (bx < (NUM_SETS + 1));

        float* data_x;
        float* data_y;
        float* data_z;
        float* random_x;
        float* random_y;
        float* random_z;

        for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
        {
            if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
            {
                warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
            }
        }

        // Get stuff into shared memory to kick off the loop.
        if( !do_self)
        {
            data_x = all_x_data;
            data_y = all_y_data;
            data_z = all_z_data;
            random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        }
        else
        {
            random_x = all_x_data + NUM_ELEMENTS * (bx);
            random_y = all_y_data + NUM_ELEMENTS * (bx);
            random_z = all_z_data + NUM_ELEMENTS * (bx);
            
            data_x = random_x;
            data_y = random_y;
            data_z = random_z;
        }

        // Iterate over all data points
        for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
        {
            // load current set of data into shared memory
            // (total of BLOCK_SIZE points loaded)
            if( tid + i < NUM_ELEMENTS )
            { // reading outside of bounds is a-okay
                data_s[tid] = (struct cartesian)
                        {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
            }
            
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

            // Iterate over all random points
            for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; j += BLOCK_SIZE)
            {
                // load current random point values
                float random_x_s;
                float random_y_s;
                float random_z_s;
                
                if(tid + j < NUM_ELEMENTS)
                {
                    random_x_s = random_x[tid + j];
                    random_y_s = random_y[tid + j];
                    random_z_s = random_z[tid + j];
                }

                // Iterate for all elements of current set of data points 
                // (BLOCK_SIZE iterations per thread)
                // Each thread calcs against 1 random point within cur set of random
                // (so BLOCK_SIZE threads covers all random points within cur set)
                for(unsigned int k = 0; 
                    (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                    k += 1)
                {
                    // do actual calculations on the values:
                    float distance = 
                        data_s[k].x * random_x_s +
                        data_s[k].y * random_y_s +
                        data_s[k].z * random_z_s;

                    unsigned int bin_index;

                    // run binary search to find bin_index
                    unsigned int min = 0;
                    unsigned int max = NUM_BINS;
                    {
                        unsigned int k2;
                        
                        while (max > min+1)
                        {
                            k2 = (min + max) / 2;
                            if (distance >= dev_binb[k2]) 
                            max = k2;
                            else 
                            min = k2;
                        }
                        bin_index = max - 1;
                    }

                    unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                    if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                    (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
                    {
                        atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                    }
                }
            }
            // TODO: why no sync?
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
        }

        // coalesce the histograms in a block
        unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
        offset >>= 1)
        {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
            bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
            {
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                {
                    unsigned long sum =
                    warp_hists[bin_base + bin_index][warp_index] + 
                    warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        // __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = histograms + NUM_BINS * bx;
        if(tid < NUM_BINS)
        {
            hist_base[tid] = warp_hists[tid][0];
        }
        // TODO: __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
    }
}

__device__ void general_ptb_tpacf0(hist_t* histograms, float* all_x_data, float* all_y_data, float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;


    // __shared__ struct cartesian data_s[BLOCK_SIZE];
    // TODO : in mix version, why no shared? 
    __shared__ struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        unsigned int bx = block_id_x;
        unsigned int tid = thread_id_x;
        bool do_self = (bx < (NUM_SETS + 1));

        float* data_x;
        float* data_y;
        float* data_z;
        float* random_x;
        float* random_y;
        float* random_z;

        for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
        {
            if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
            {
                warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
            }
        }

        // Get stuff into shared memory to kick off the loop.
        if( !do_self)
        {
            data_x = all_x_data;
            data_y = all_y_data;
            data_z = all_z_data;
            random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        }
        else
        {
            random_x = all_x_data + NUM_ELEMENTS * (bx);
            random_y = all_y_data + NUM_ELEMENTS * (bx);
            random_z = all_z_data + NUM_ELEMENTS * (bx);
            
            data_x = random_x;
            data_y = random_y;
            data_z = random_z;
        }

        // Iterate over all data points
        for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
        {
            // load current set of data into shared memory
            // (total of BLOCK_SIZE points loaded)
            if( tid + i < NUM_ELEMENTS )
            { // reading outside of bounds is a-okay
                data_s[tid] = (struct cartesian)
                        {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
            }
            
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

            // Iterate over all random points
            for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; j += BLOCK_SIZE)
            {
                // load current random point values
                float random_x_s;
                float random_y_s;
                float random_z_s;
                
                if(tid + j < NUM_ELEMENTS)
                {
                    random_x_s = random_x[tid + j];
                    random_y_s = random_y[tid + j];
                    random_z_s = random_z[tid + j];
                }

                // Iterate for all elements of current set of data points 
                // (BLOCK_SIZE iterations per thread)
                // Each thread calcs against 1 random point within cur set of random
                // (so BLOCK_SIZE threads covers all random points within cur set)
                for(unsigned int k = 0; 
                    (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                    k += 1)
                {
                    // do actual calculations on the values:
                    float distance = 
                        data_s[k].x * random_x_s +
                        data_s[k].y * random_y_s +
                        data_s[k].z * random_z_s;

                    unsigned int bin_index;

                    // run binary search to find bin_index
                    unsigned int min = 0;
                    unsigned int max = NUM_BINS;
                    {
                        unsigned int k2;
                        
                        while (max > min+1)
                        {
                            k2 = (min + max) / 2;
                            if (distance >= dev_binb[k2]) 
                            max = k2;
                            else 
                            min = k2;
                        }
                        bin_index = max - 1;
                    }

                    unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                    if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                    (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
                    {
                        atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                    }
                }
            }
            // TODO: why no sync?
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
        }

        // coalesce the histograms in a block
        unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
        offset >>= 1)
        {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
            bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
            {
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                {
                    unsigned long sum =
                    warp_hists[bin_base + bin_index][warp_index] + 
                    warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        // __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = histograms + NUM_BINS * bx;
        if(tid < NUM_BINS)
        {
            hist_base[tid] = warp_hists[tid][0];
        }
        // TODO: __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
    }
}

__device__ void general_ptb_tpacf0(hist_t* histograms, float* all_x_data, float* all_y_data, float* all_z_data, int NUM_SETS, int NUM_ELEMENTS,
	    int grid_dimension_x, int grid_dimension_y, int grid_dimension_z, int block_dimension_x, int block_dimension_y, int block_dimension_z,  
		    int ptb_start_block_pos, int ptb_iter_block_step, int ptb_end_block_pos, int thread_base) {

    // ori
    // unsigned int block_pos = blockIdx.x;
    // int thread_id_x = (threadIdx.x - thread_step) % block_dimension_x;
    // int thread_id_y = (threadIdx.x - thread_step) / block_dimension_x;

    unsigned int block_pos = blockIdx.x + ptb_start_block_pos;

    int thread_id_x = (threadIdx.x - thread_base) % block_dimension_x;
    // int thread_id_y = ((threadIdx.x - thread_base) / block_dimension_x) % block_dimension_y;


    // __shared__ struct cartesian data_s[BLOCK_SIZE];
    // TODO : in mix version, why no shared? 
    __shared__ struct cartesian data_s[BLOCK_SIZE];
    __shared__ unsigned int warp_hists[NUM_BINS][NUM_HISTOGRAMS];
    // 640B <1k  

    for (;; block_pos += ptb_iter_block_step) {
        if (block_pos >= ptb_end_block_pos) {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
		// int block_id_y = (block_pos / grid_dimension_x) % grid_dimension_y;

        unsigned int bx = block_id_x;
        unsigned int tid = thread_id_x;
        bool do_self = (bx < (NUM_SETS + 1));

        float* data_x;
        float* data_y;
        float* data_z;
        float* random_x;
        float* random_y;
        float* random_z;

        for(unsigned int w = 0; w < NUM_BINS*NUM_HISTOGRAMS; w += BLOCK_SIZE )
        {
            if(w+tid < NUM_BINS*NUM_HISTOGRAMS)
            {
                warp_hists[(w+tid)/NUM_HISTOGRAMS][(w+tid)%NUM_HISTOGRAMS] = 0;
            }
        }

        // Get stuff into shared memory to kick off the loop.
        if( !do_self)
        {
            data_x = all_x_data;
            data_y = all_y_data;
            data_z = all_z_data;
            random_x = all_x_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_y = all_y_data + NUM_ELEMENTS * (bx - NUM_SETS);
            random_z = all_z_data + NUM_ELEMENTS * (bx - NUM_SETS);
        }
        else
        {
            random_x = all_x_data + NUM_ELEMENTS * (bx);
            random_y = all_y_data + NUM_ELEMENTS * (bx);
            random_z = all_z_data + NUM_ELEMENTS * (bx);
            
            data_x = random_x;
            data_y = random_y;
            data_z = random_z;
        }

        // Iterate over all data points
        for(unsigned int i = 0; i < NUM_ELEMENTS; i += BLOCK_SIZE )
        {
            // load current set of data into shared memory
            // (total of BLOCK_SIZE points loaded)
            if( tid + i < NUM_ELEMENTS )
            { // reading outside of bounds is a-okay
                data_s[tid] = (struct cartesian)
                        {data_x[tid + i], data_y[tid + i], data_z[tid + i]};
            }
            
            // __syncthreads();
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

            // Iterate over all random points
            for(unsigned int j = (do_self ? i+1 : 0); j < NUM_ELEMENTS; j += BLOCK_SIZE)
            {
                // load current random point values
                float random_x_s;
                float random_y_s;
                float random_z_s;
                
                if(tid + j < NUM_ELEMENTS)
                {
                    random_x_s = random_x[tid + j];
                    random_y_s = random_y[tid + j];
                    random_z_s = random_z[tid + j];
                }

                // Iterate for all elements of current set of data points 
                // (BLOCK_SIZE iterations per thread)
                // Each thread calcs against 1 random point within cur set of random
                // (so BLOCK_SIZE threads covers all random points within cur set)
                for(unsigned int k = 0; 
                    (k < BLOCK_SIZE) && (k+i < NUM_ELEMENTS);
                    k += 1)
                {
                    // do actual calculations on the values:
                    float distance = 
                        data_s[k].x * random_x_s +
                        data_s[k].y * random_y_s +
                        data_s[k].z * random_z_s;

                    unsigned int bin_index;

                    // run binary search to find bin_index
                    unsigned int min = 0;
                    unsigned int max = NUM_BINS;
                    {
                        unsigned int k2;
                        
                        while (max > min+1)
                        {
                            k2 = (min + max) / 2;
                            if (distance >= dev_binb[k2]) 
                            max = k2;
                            else 
                            min = k2;
                        }
                        bin_index = max - 1;
                    }

                    unsigned int warpnum = tid / (WARP_SIZE/HISTS_PER_WARP);
                    if((distance < dev_binb[min]) && (distance >= dev_binb[max]) && 
                    (!do_self || (tid + j > i + k)) && (tid + j < NUM_ELEMENTS))
                    {
                        atomicAdd(&warp_hists[bin_index][warpnum], 1U);
                    }
                }
            }
            // TODO: why no sync?
            asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
        }

        // coalesce the histograms in a block
        unsigned int warp_index = tid & ( (NUM_HISTOGRAMS>>1) - 1);
        unsigned int bin_index = tid / (NUM_HISTOGRAMS>>1);
        for(unsigned int offset = NUM_HISTOGRAMS >> 1; offset > 0; 
        offset >>= 1)
        {
            for(unsigned int bin_base = 0; bin_base < NUM_BINS; 
            bin_base += BLOCK_SIZE/ (NUM_HISTOGRAMS>>1))
            {
                // __syncthreads();
                asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

                if(warp_index < offset && bin_base+bin_index < NUM_BINS )
                {
                    unsigned long sum =
                    warp_hists[bin_base + bin_index][warp_index] + 
                    warp_hists[bin_base + bin_index][warp_index+offset];
                    warp_hists[bin_base + bin_index][warp_index] = sum;
                }
            }
        }

        // __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");

        // Put the results back in the real histogram
        // warp_hists[x][0] holds sum of all locations of bin x
        hist_t* hist_base = histograms + NUM_BINS * bx;
        if(tid < NUM_BINS)
        {
            hist_base[tid] = warp_hists[tid][0];
        }
        // TODO: __syncthreads();
        asm volatile("bar.sync %0, %1;" : : "r"(2), "r"(256) : "memory");
    }
}