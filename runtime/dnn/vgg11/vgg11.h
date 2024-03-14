#pragma once
#include "Task.h"
#include "util.h"

class VGG11: public Task {
public:
    // override constructor
    VGG11(int taskId) : Task(taskId) {
        this->taskName = "VGG11";
        this->taskId = taskId;
        initParams();
    }
    VGG11(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_32_0, Parameter_32_0_host, sizeof(float) * 4816896, cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_32_0, float**  Result_99_0);
    void initParams();  
    float* Parameter_32_0;
    float** Result_99_0;
    float* Parameter_32_0_host;

};