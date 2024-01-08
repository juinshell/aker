// Task.cpp
#include "Task.h"
#include "Logger.h"

extern Logger logger;

Task::Task(int taskId, std::string taskName) : taskId(taskId), taskName(taskName){}

void Task::addKernel(std::unique_ptr<Kernel> kernel) {
    kernels.push(std::move(kernel));
}

void Task::executeTask() {
    float kernel_time = 0.0f;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));

    while (!kernels.empty()) {
        kernel_time = 0.0f;
        std::unique_ptr<Kernel> kernel = std::move(kernels.front());
        kernels.pop();

        // 执行 Kernel 的操作
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
        kernel->execute();
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
		CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
		CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        logger.INFO("kernel name: " + kernel->getKernelName() + ", kernel time: " + std::to_string(kernel_time) + " ms");
        // 函数结束，kernel析构
    }
}

Task::~Task() {
    while (!kernels.empty()) {
        kernels.pop();
    }
    logger.INFO("task name: " + taskName + ", id: " + std::to_string(taskId) + " is destroyed!");
}
