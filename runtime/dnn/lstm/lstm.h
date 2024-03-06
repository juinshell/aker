#pragma once
#include "Task.h"
#include "util.h"

class LSTM: public Task {
public:
    // override constructor
    LSTM(int taskId) : Task(taskId) {
        this->taskName = "LSTM";
        this->taskId = taskId;
        initParams();
    }
    LSTM(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void initExecution() override{
        CUDA_SAFE_CALL(cudaMemcpy(Parameter_162_0, Parameter_162_0_host, sizeof(float) * 25600, cudaMemcpyHostToDevice));
    }
    void gen_vector(float*  Parameter_162_0, float**  Result_32346_0);
    void initParams();  
    float* Parameter_162_0;
    float** Result_32346_0;
    float* Parameter_162_0_host;

};