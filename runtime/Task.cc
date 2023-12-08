// Task.cpp
#include "Task.h"

Task::Task(int taskId, std::string taskName) : taskId(taskId), taskName(taskName){}

void Task::addKernel(std::unique_ptr<Kernel> kernel) {
    kernels.push(std::move(kernel));
}

void Task::executeTask() {
    while (!kernels.empty()) {
        std::unique_ptr<Kernel> kernel = std::move(kernels.front());
        kernels.pop();

        // 执行 Kernel 的操作
        kernel->execute();
    }
}

void Task::preLaunch() {
    printf("preLaunch task -- taskId: %d, taskName:%s\n", taskId, taskName.c_str());
    return ;
}