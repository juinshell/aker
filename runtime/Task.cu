// Task.cpp
#include "Task.h"
#include "Logger.h"
#include "Recorder.h"

extern Logger logger;

extern Recorder recorder;

Task::Task(int taskId, std::string taskName) : taskId(taskId), taskName(taskName){}
Task::Task(int taskId) : taskId(taskId) {}

void Task::addKernel(std::unique_ptr<Kernel> kernel) {
    kernels.emplace_back(std::move(kernel));
}

void Task::executeTask(ExecutionMode mode) {
    float kernel_time = 0.0f;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));

    for (auto &kernel : kernels) {
        logger.INFO("kernel name: " + kernel->kernelName + ", id: " + std::to_string(kernel->Id) + " is executing ...");

        // execute kernel
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
        kernel->execute();
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        logger.DEBUG("kernel name: " + kernel->getKernelName() + ", kernel time: " + std::to_string(kernel_time) + " ms");

        // record kernel time
        if (mode == ExecutionMode::PROFILE)
            recorder.recordKernel(taskId, kernel->Id, kernel->kernelName, kernel_time);
        // ~kernel
    }
    // record task
    if (mode == ExecutionMode::PROFILE)
    recorder.recordTask(taskId, taskName);
    // logger.INFO("task name: " + taskName + ", id: " + std::to_string(taskId) + " is executed!");
}

Task::~Task() {
    std::vector<std::unique_ptr<Kernel>>().swap(kernels);
    // logger.INFO("task name: " + taskName + ", id: " + std::to_string(taskId) + " is destroyed!");
}
