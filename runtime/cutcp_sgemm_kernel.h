/*** 
 * @Author: diagonal
 * @Date: 2023-12-08 21:52:35
 * @LastEditors: diagonal
 * @LastEditTime: 2023-12-09 12:37:13
 * @FilePath: /tacker/runtime/cp_kernel.h
 * @Description: 
 * @happy coding, happy life!
 * @Copyright (c) 2023 by jxdeng, All Rights Reserved. 
 */
// cp_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"

struct CUTCP_SGEMM_ParamsStruct {
    int cutcp0_binDim_x;
    int cutcp0_binDim_y;
    float4* cutcp0_binZeroAddr;
    float cutcp0_h;
    float cutcp0_cutoff2;
    float cutcp0_inv_cutoff2;
    float* cutcp0_regionZeroAddr;
    int cutcp0_zRegionIndex_t;
    int cutcp0_grid_dimension_x;
    int cutcp0_grid_dimension_y;
    int cutcp0_grid_dimension_z;
    int cutcp0_block_dimension_x;
    int cutcp0_block_dimension_y;
    int cutcp0_block_dimension_z;
    int cutcp0_ptb_start_block_pos;
    int cutcp0_ptb_iter_block_step;
    int cutcp0_ptb_end_block_pos;
    float* sgemm1_A;
    float* sgemm1_B;
    float* sgemm1_C;
    int sgemm1_NORMAL_M;
    int sgemm1_NORMAL_N;
    int sgemm1_NORMAL_K;
    int sgemm1_grid_dimension_x;
    int sgemm1_grid_dimension_y;
    int sgemm1_grid_dimension_z;
    int sgemm1_block_dimension_x;
    int sgemm1_block_dimension_y;
    int sgemm1_block_dimension_z;
    int sgemm1_ptb_start_block_pos;
    int sgemm1_ptb_iter_block_step;
    int sgemm1_ptb_end_block_pos;
};

class CUTCP_SGEMM_Kernel : public Kernel {
public:
    // 构造函数
    CUTCP_SGEMM_Kernel(int id, const std::string& name);
    CUTCP_SGEMM_Kernel(int id, const std::string& name, CUTCP_SGEMM_ParamsStruct& params);

    // 析构函数
    ~CUTCP_SGEMM_Kernel();

    // 实现纯虚函数
    void execute();
    void initParams();

private:
    void loadKernel();  
    CUTCP_SGEMM_ParamsStruct* kernelParams;
    
};