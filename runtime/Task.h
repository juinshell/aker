// Task.h
#pragma once
#include "Kernel.h"
#include <queue>

class Task {
public:
    Task(int taskId, std::string taskName);
    void addKernel(std::unique_ptr<Kernel> kernel);
    void preLaunch();
    void executeTask();

private:
    int taskId;
    std::string taskName;
    std::queue<std::unique_ptr<Kernel>> kernels;
};