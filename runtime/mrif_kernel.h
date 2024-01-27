//MRIF_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


struct OriMRIFParamsStruct {
    int numK;
    int kGlobalIndex;
    float* x;
    float* y;
    float* z;
    float* outR;
    float* outI;
};

class OriMRIFKernel : public Kernel {
public:
    // 构造函数
    OriMRIFKernel(int id);

    // 析构函数
    ~OriMRIFKernel();

    // 实现纯虚函数
    void executeImpl();
    void initParams();

private:
    void loadKernel(){};  
    OriMRIFParamsStruct* MRIFKernelParams;
    
};