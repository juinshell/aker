// fft_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"

struct OriFFTParamsStruct {
    float2 *data;
};

class OriFFTKernel : public Kernel {
public:
    // 构造函数
    OriFFTKernel(int id);
    OriFFTKernel(int id, const std::string& moduleName, const std::string& kernelName);

    // 析构函数
    ~OriFFTKernel();

    // 实现纯虚函数
    void executeImpl();
    void initParams();
    std::vector<int> getArgs() {
        return std::vector<int>();
    }

private:
    void loadKernel();  
    OriFFTParamsStruct* FFTKernelParams;
    
};