// Kernel.h
#pragma once

#include "util.h"

class Kernel {
public:
    virtual void execute() = 0;
    virtual void initParams() = 0;
    virtual void loadKernel() = 0;

    int getKernelId() { return kernelId; }
    std::string& getKernelName() { return kernelName; }


protected:
    int kernelId;
    std::string kernelName;
    CUfunction function;
    CUmodule module;
    dim3 launchGridDim;
    dim3 launchBlockDim;
};