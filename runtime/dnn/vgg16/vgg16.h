#pragma once
#include "Task.h"
#include "util.h"

class VGG16: public Task {
public:
    // override constructor
    VGG16(int taskId) : Task(taskId) {
        this->taskName = "VGG16";
        this->taskId = taskId;
        initParams();
    }
    VGG16(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(float) * 2408448, cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_32_0, float**  Result_99_0);
    void initParams();  
    float* Parameter_0_0;
    float** Result_144_0;
    float* Parameter_0_0_host;

};