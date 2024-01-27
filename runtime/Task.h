// Task.h
#pragma once
#include "Kernel.h"
#include <queue>

enum ExecutionMode{
    WARMUP = 0,
    PROFILE
};

class Task {
public:
    Task(int taskId, std::string taskName);
    Task(int taskId);
    ~Task();
    void addKernel(std::unique_ptr<Kernel> kernel);
    void executeTask(ExecutionMode mode);

    int taskId;
    std::string taskName;
    std::vector<std::unique_ptr<Kernel>> kernels; // 容器一般不能存放引用
};