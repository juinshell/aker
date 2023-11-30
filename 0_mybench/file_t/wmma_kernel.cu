
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void ori_wmma(half *a, half *b, float *c, int MATRIX_M, int MATRIX_N, int MATRIX_K, int iteration) {
    // Leading dimensions. Packed with no transpositions.
    int lda = MATRIX_M;
    int ldb = MATRIX_K;
    int ldc = MATRIX_M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int loop = 0; loop < iteration; loop++) 
    {
        for (int i = 0; i < MATRIX_K; i += WMMA_K)
        {
            int aRow = warpM * WMMA_M;
            int aCol = i;
            int bRow = i;
            int bCol = warpN * WMMA_N;

            // Bounds checking
            if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N)
            {
                // Load the inputs
                // ptx asm("load. bypass_l1cache")
                wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
                wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
                
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }

        // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
        int cRow = warpM * WMMA_M;
        int cCol = warpN * WMMA_N;

        if (cRow < MATRIX_M && cCol < MATRIX_N) {
            wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
            for(int i=0; i < c_frag.num_elements; i++) {
                c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
            }
            wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
        }
    }
}


__global__ void pers_wmma(half *a, half *b, float *c, int MATRIX_M, int MATRIX_N, int MATRIX_K,
                int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
   // Leading dimensions. Packed with no transpositions.
    int lda = MATRIX_M;
    int ldb = MATRIX_K;
    int ldc = MATRIX_M;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    int thread_id_y = threadIdx.x / block_dimension_x;

    for (;; block_pos += gridDim.x) {
        if (block_pos >= grid_dimension_x * grid_dimension_y)
        {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        // Tile using a 2D grid
        int warpM = (block_id_x * block_dimension_x + thread_id_x) / warpSize;
        int warpN = block_id_y * block_dimension_y + thread_id_y;

        wmma::fill_fragment(acc_frag, 0.0f);

        for (int loop = 0; loop < iteration; loop++) 
        {
            for (int i = 0; i < MATRIX_K; i += WMMA_K)
            {
                int aRow = warpM * WMMA_M;
                int aCol = i;
                int bRow = i;
                int bCol = warpN * WMMA_N;

                // Bounds checking
                if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N)
                {
                    // Load the inputs
                    // ptx asm("load. bypass_l1cache")
                    wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
                    wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
                    // for (int t = 0; t < 10000; t++)
                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
            }

            // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
            int cRow = warpM * WMMA_M;
            int cCol = warpN * WMMA_N;

            if (cRow < MATRIX_M && cCol < MATRIX_N) {
                wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
                for(int i=0; i < c_frag.num_elements; i++) {
                    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                }
                wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
            }
        }
    }
}


__device__ void mix_wmma(half *a, half *b, float *c, int MATRIX_M, int MATRIX_N, int MATRIX_K,
                int grid_dimension_x, int grid_dimension_y, int block_dimension_x, int block_dimension_y, int iteration) {
   // Leading dimensions. Packed with no transpositions.
    int lda = MATRIX_M;
    int ldb = MATRIX_K;
    int ldc = MATRIX_M;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    unsigned int block_pos = blockIdx.x;
    int thread_id_x = threadIdx.x % block_dimension_x;
    int thread_id_y = threadIdx.x / block_dimension_x;

    for (;; block_pos += WMMA_GRID_DIM) {
        if (block_pos >= grid_dimension_x * grid_dimension_y)
        {
            return;
        }

        int block_id_x = block_pos % grid_dimension_x;
        int block_id_y = block_pos / grid_dimension_x;

        // Tile using a 2D grid
        int warpM = (block_id_x * block_dimension_x + thread_id_x) / warpSize;
        int warpN = block_id_y * block_dimension_y + thread_id_y;

        wmma::fill_fragment(acc_frag, 0.0f);

        for (int loop = 0; loop < iteration; loop++) 
        {
            for (int i = 0; i < MATRIX_K; i += WMMA_K)
            {
                int aRow = warpM * WMMA_M;
                int aCol = i;
                int bRow = i;
                int bCol = warpN * WMMA_N;

                // Bounds checking
                if (aRow < MATRIX_M && aCol < MATRIX_K && bRow < MATRIX_K && bCol < MATRIX_N)
                {
                    // Load the inputs
                    // ptx asm("load. bypass_l1cache")
                    wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
                    wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
                    // for (int t = 0; t < 10000; t++)
                    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                }
            }

            // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
            int cRow = warpM * WMMA_M;
            int cCol = warpN * WMMA_N;

            if (cRow < MATRIX_M && cCol < MATRIX_N) {
                wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
                for(int i=0; i < c_frag.num_elements; i++) {
                    c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
                }
                wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
            }
        }
    }
}


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}
