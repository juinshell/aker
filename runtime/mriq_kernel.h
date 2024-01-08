//mriq_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


struct OriMRIQParamsStruct {
    int numK;
    int kGlobalIndex;
    float* x;
    float* y;
    float* z;
    float* Qr;
    float* Qi;
};

class OriMRIQKernel : public Kernel {
public:
    // 构造函数
    OriMRIQKernel(int id);

    // 析构函数
    ~OriMRIQKernel();

    // 实现纯虚函数
    void execute();
    void initParams();

private:
    void loadKernel(){};  
    OriMRIQParamsStruct* MRIQKernelParams;
    
};