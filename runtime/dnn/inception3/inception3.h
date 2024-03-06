#pragma once
#include "Task.h"
#include "util.h"

class Inception3: public Task {
public:
    // override constructor
    Inception3(int taskId) : Task(taskId) {
        this->taskName = "Inception3";
        this->taskId = taskId;
        initParams();
    }
    Inception3(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_485_0, Parameter_485_0_host, sizeof(float) * 268203, cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_270_0, float**  Result_505_0);
    void initParams();  
    float* Parameter_485_0;
    float** Result_892_0;
    float* Parameter_485_0_host;

};