#pragma once
#include "Task.h"
#include "util.h"

class Resnet50: public Task {
public:
    // override constructor
    Resnet50(int taskId) : Task(taskId) {
        this->taskName = "Resnet50";
        this->taskId = taskId;
        initParams();
    }
    Resnet50(int taskId, std::string taskName) : Task(taskId, taskName) {
        this->taskName = taskName;
        this->taskId = taskId;
        initParams();
    }
    void gen_vector(float*  Parameter_270_0, float**  Result_505_0);
private:
    void initParams();  
    float* Parameter_270_0;
    float** Result_505_0;

};