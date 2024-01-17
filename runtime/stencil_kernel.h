//mriq_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"


struct OriSTENCILParamsStruct {
    float c0;
    float c1;
    float *A0;
    float *Anext;
    int nx;
    int ny;
    int nz;
};

class OriSTENCILKernel : public Kernel {
public:
    // 构造函数
    OriSTENCILKernel(int id);

    // 析构函数
    ~OriSTENCILKernel();

    // 实现纯虚函数
    void execute();
    void initParams();

private:
    void loadKernel(){};  
    OriSTENCILParamsStruct* STENCILKernelParams;
    
};