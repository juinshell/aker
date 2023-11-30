void readImage(const char *fName, unsigned int *hh_DataA, unsigned DATA_SIZE)
{

    FILE *File;
    unsigned int temp;
    int y;

    if((File = fopen(fName, "rb")) != NULL) {
        for (y=0; y < DATA_SIZE; y++){
            int fr = fread(&temp, sizeof(unsigned int), 1, File);
            hh_DataA[y] = (unsigned int)ByteSwap16(temp);
            if(hh_DataA[y] >= 4096) {
                hh_DataA[y] = 4095;
            }
        }
        fclose(File);
    } else {
        printf("%s does not exist\n", fName);
        exit(1);
    }
}

__global__ void ori_img(unsigned int* histo,
                            unsigned int* data,
                            int size, int BINS, 
                            int iteration)
{
    // Block and thread index
    const int bx = blockIdx.x;
    const int tx = threadIdx.x;
    // Warp and lane
    const unsigned int warpid = tx >> 5;
    const unsigned int lane = tx & 31;	

    __shared__ unsigned int Hs[BINSp * REP];

    // Offset to per-block sub-histograms
    const unsigned int off_rep = BINSp * (tx % REP);

    // Constants for interleaved read access
    const int warps_block = blockDim.x / WARP_SIZE;
    const int begin = (size / warps_block) * warpid + WARP_SIZE * bx + lane;
    const int end = (size / warps_block) * (warpid + 1);
    const int step = WARP_SIZE * gridDim.x;

    for (int loop = 0; loop < iteration; loop++) {
        // Sub-histograms initialization
        for(int pos = tx; pos < BINSp*REP; pos += blockDim.x) {
            Hs[pos] = 0;
        }

        __syncthreads();	// Intra-block synchronization

        // Main loop
        for(int i = begin; i < end; i += step){
            // Global memory read
            unsigned int d = data[i];

            // Atomic vote in shared memory
            atomicAdd(&Hs[off_rep + ((d * BINS) >> 12)], 1);
        }

        __syncthreads();	// Intra-block synchronization

        // Merge per-block histograms and write to global memory
        for(int pos = tx; pos < BINS; pos += blockDim.x){
            unsigned int sum = 0;
            for(int base = 0; base < BINSp*REP; base += BINSp)
                sum += Hs[base + pos];
            // Atomic addition in global memory
            atomicAdd(histo + pos, sum);
        }
        __syncthreads();	// Intra-block synchronization
    }
}


__global__ void ptb_img(unsigned int* histo, unsigned int* data,
                        int size, int BINS, int grid_dimension_x, int block_dimension_x,
                        int iteration)
{
    unsigned int block_pos = blockIdx.x;
    const int tx = threadIdx.x;
    // Warp and lane
    const unsigned int warpid = tx >> 5;
    const unsigned int lane = tx & 31;	
    // Offset to per-block sub-histograms
    const unsigned int off_rep = BINSp * (tx % REP);

    __shared__ unsigned int Hs[BINSp * REP];

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x) {
            return;
        }

        // Block and thread index
        int bx = block_pos;    

        // Constants for interleaved read access
        int warps_block = block_dimension_x / WARP_SIZE;
        int begin = (size / warps_block) * warpid + WARP_SIZE * bx + lane;
        int end = (size / warps_block) * (warpid + 1);
        int step = WARP_SIZE * grid_dimension_x;

        for (int loop = 0; loop < iteration; loop++) {
            // Sub-histograms initialization
            for(int pos = tx; pos < BINSp*REP; pos += block_dimension_x) {
                Hs[pos] = 0;
            }

            __syncthreads();	// Intra-block synchronization

            // Main loop
            for(int i = begin; i < end; i += step){
                // Global memory read
                unsigned int d = data[i];

                // Atomic vote in shared memory
                atomicAdd(&Hs[off_rep + ((d * BINS) >> 12)], 1);
            }

            __syncthreads();	// Intra-block synchronization

            // Merge per-block histograms and write to global memory
            for(int pos = tx; pos < BINS; pos += block_dimension_x){
                unsigned int sum = 0;
                for(int base = 0; base < BINSp*REP; base += BINSp)
                    sum += Hs[base + pos];
                // Atomic addition in global memory
                atomicAdd(histo + pos, sum);
            }
            __syncthreads();	// Intra-block synchronization
        }
    }
}
