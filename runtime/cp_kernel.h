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

struct OriCPParamsStruct {
    int numatoms;
    float gridspacing;
    float * energygrid;
};

class OriCPKernel : public Kernel {
public:
    // 构造函数
    OriCPKernel(int id);
    OriCPKernel(int id, const std::string& moduleName, const std::string& kernelName);

    // 析构函数
    ~OriCPKernel();

    // 实现纯虚函数
    void executeImpl(cudaStream_t stream);
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel();  
    OriCPParamsStruct* CPKernelParams;
    
};

