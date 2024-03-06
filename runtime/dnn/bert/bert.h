#pragma once
#include "Task.h"
#include "util.h"

class Bert: public Task {
public:
    // override constructor
    Bert(int taskId) : Task(taskId) {
        this->taskName = "Bert";
        this->taskId = taskId;
        initParams();
    }
    Bert(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_964_0, Parameter_964_0_host, sizeof(int32_t) * 512, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_965_0, Parameter_965_0_host, sizeof(int32_t) * 512, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_966_0, Parameter_966_0_host, sizeof(int32_t) * 512, cudaMemcpyHostToDevice));
    }
    void gen_vector(int32_t*  Parameter_964_0, int32_t*  Parameter_965_0, int32_t*  Parameter_966_0, float**  Result_3689_0);
    void initParams();  
    int32_t*  Parameter_964_0;
    int32_t* Parameter_964_0_host;
    int32_t*  Parameter_965_0;
    int32_t*  Parameter_965_0_host;
    int32_t*  Parameter_966_0;
    int32_t*  Parameter_966_0_host;
    float**  Result_3689_0;

};