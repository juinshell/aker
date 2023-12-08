// cp_kernel.h
#pragma once
#include "Kernel.h"
#include "util.h"

struct OriCPParamsStruct {
    int numatoms;
    float gridspacing;
    float * energygrid;
    int iteration;
};

class OriCPKernel : public Kernel {
public:
    // 构造函数
    OriCPKernel(int id, const std::string& name);
    OriCPKernel(int id, const std::string& name, OriCPParamsStruct& params);

    // 析构函数
    ~OriCPKernel();

    // 实现纯虚函数
    void execute();
    void initParams();

private:
    void loadKernel();  
    OriCPParamsStruct* kernelParams;
    
};