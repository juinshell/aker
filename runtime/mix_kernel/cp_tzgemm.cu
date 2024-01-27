__global__ void mix_kernel1(
    half *a, half *b, float *c,
	int MATRIX_M, int MATRIX_N, int MATRIX_K,
    int wmma_grid_dim_x, int wmma_block_dim_x, 
    int wmma_iter,
    int numatoms, float gridspacing, float * energygrid, 
	int cp_grid_dim_x, int cp_grid_dim_y, int cp_block_dim_x, int cp_block_dim_y,
	int cp_iter){
    if (threadIdx.x < 128 * 1) {
        mix_tzgemm0(a, b, c, 
			MATRIX_M, MATRIX_N, MATRIX_K,
			wmma_grid_dim_x, 128, wmma_iter);
    } else {
        mix_cp0(numatoms, gridspacing, energygrid, 
			cp_grid_dim_x, cp_grid_dim_y, cp_block_dim_x, cp_block_dim_y, 128,
			cp_iter);
    }
}