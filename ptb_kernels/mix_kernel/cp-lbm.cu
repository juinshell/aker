// cp-lbm-6-1
__global__ void mixed_cp_lbm_kernel_6_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cp3(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 3 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_cp4(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 4 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_cp5(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 5 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 768
        );
    }

}

// cp-lbm-6-2
__global__ void mixed_cp_lbm_kernel_6_2(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cp3(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 3 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_cp4(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 4 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_cp5(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 5 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 6, cp0_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 2, lbm1_ptb_end_block_pos, 768
        );
    }
    else if (threadIdx.x < 1024) {
        general_ptb_lbm1(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 1 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 2, lbm1_ptb_end_block_pos, 896
        );
    }

}

// cp-lbm-3-1
__global__ void mixed_cp_lbm_kernel_3_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 3, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 3, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 3, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 384
        );
    }

}

// cp-lbm-1-1
__global__ void mixed_cp_lbm_kernel_1_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 1, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 128
        );
    }

}

// cp-lbm-2-1
__global__ void mixed_cp_lbm_kernel_2_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 2, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 256
        );
    }

}

// cp-lbm-4-1
__global__ void mixed_cp_lbm_kernel_4_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 4, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 4, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 4, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cp3(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 3 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 4, cp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 512
        );
    }

}

// cp-lbm-5-1
__global__ void mixed_cp_lbm_kernel_5_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 5, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 5, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 5, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cp3(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 3 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 5, cp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_cp4(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 4 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 5, cp0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 640
        );
    }

}

// cp-lbm-7-1
__global__ void mixed_cp_lbm_kernel_7_1(int cp0_numatoms, float cp0_gridspacing, float* cp0_energygrid, int cp0_grid_dimension_x, int cp0_grid_dimension_y, int cp0_grid_dimension_z, int cp0_block_dimension_x, int cp0_block_dimension_y, int cp0_block_dimension_z, int cp0_ptb_start_block_pos, int cp0_ptb_iter_block_step, int cp0_ptb_end_block_pos, float* lbm1_srcGrid, float* lbm1_dstGrid, int lbm1_grid_dimension_x, int lbm1_grid_dimension_y, int lbm1_grid_dimension_z, int lbm1_block_dimension_x, int lbm1_block_dimension_y, int lbm1_block_dimension_z, int lbm1_ptb_start_block_pos, int lbm1_ptb_iter_block_step, int lbm1_ptb_end_block_pos){
    if (threadIdx.x < 128) {
        general_ptb_cp0(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 0 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 7, cp0_ptb_end_block_pos, 0
        );
    }
    else if (threadIdx.x < 256) {
        general_ptb_cp1(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 1 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 7, cp0_ptb_end_block_pos, 128
        );
    }
    else if (threadIdx.x < 384) {
        general_ptb_cp2(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 2 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 7, cp0_ptb_end_block_pos, 256
        );
    }
    else if (threadIdx.x < 512) {
        general_ptb_cp3(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 3 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 7, cp0_ptb_end_block_pos, 384
        );
    }
    else if (threadIdx.x < 640) {
        general_ptb_cp4(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 4 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 7, cp0_ptb_end_block_pos, 512
        );
    }
    else if (threadIdx.x < 768) {
        general_ptb_cp5(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 5 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 7, cp0_ptb_end_block_pos, 640
        );
    }
    else if (threadIdx.x < 896) {
        general_ptb_cp6(
            cp0_numatoms, cp0_gridspacing, cp0_energygrid, cp0_grid_dimension_x, cp0_grid_dimension_y, cp0_grid_dimension_z, cp0_block_dimension_x, cp0_block_dimension_y, cp0_block_dimension_z, cp0_ptb_start_block_pos + 6 * cp0_ptb_iter_block_step, cp0_ptb_iter_block_step * 7, cp0_ptb_end_block_pos, 768
        );
    }
    else if (threadIdx.x < 1024) {
        general_ptb_lbm0(
            lbm1_srcGrid, lbm1_dstGrid, lbm1_grid_dimension_x, lbm1_grid_dimension_y, lbm1_grid_dimension_z, lbm1_block_dimension_x, lbm1_block_dimension_y, lbm1_block_dimension_z, lbm1_ptb_start_block_pos + 0 * lbm1_ptb_iter_block_step, lbm1_ptb_iter_block_step * 1, lbm1_ptb_end_block_pos, 896
        );
    }

}

