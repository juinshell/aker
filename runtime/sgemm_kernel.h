//mriq_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


struct OriSGEMMParamsStruct {
    float *A;
    float *B;
    float *C;
    int NORMAL_M;
    int NORMAL_N;
    int NORMAL_K;
};

class OriSGEMMKernel : public Kernel {
public:
    // 构造函数
    OriSGEMMKernel(int id);

    // 析构函数
    ~OriSGEMMKernel();

    // 实现纯虚函数
    void execute();
    void initParams();

private:
    void loadKernel(){};  
    OriSGEMMParamsStruct* SGEMMKernelParams;
    
};