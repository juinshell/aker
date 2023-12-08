// main.cc
#include "TackerConfig.h"
#include "util.h"
#include "TaskManager.h"
#include "Task.h"
#include "Kernel.h"
#include "Logger.h"
#include "cp_kernel.h"

Logger logger(LOG_FILE_PATH, true, true);

void compileInfo() {
    std::cout << "Tacker Version: " + std::to_string(Tacker_VERSION_MAJOR) + "." + std::to_string(Tacker_VERSION_MINOR) + "." + std::to_string(Tacker_VERSION_PATCH) << std::endl;
    std::cout << "Compile Timestamp: " + std::string(COMPILE_TIMESTAMP) << std::endl;
}

void initCUDA(int device=0) {
    int deviceCount;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        logger.ERROR("No CUDA-compatible devices found.");
        exit(EXIT_FAILURE);
    }

    CUDA_SAFE_CALL(cudaSetDevice(device));

    CUDA_SAFE_CALL(cudaDeviceReset()); // Reset device state
    CUDA_SAFE_CALL(cudaSetDevice(device)); // Set the current device

    CUDA_SAFE_CALL(cudaFree(0)); // Create a CUDA context

    logger.INFO("CUDA init complete");
}

void printDeviceProp() {
    // uses cuda runtime API
    int SMnum, blocknum, threads, warp, kernel, overlap, sharedmemory;
    cudaDeviceGetAttribute(&SMnum, cudaDevAttrMultiProcessorCount, 0);
    cudaDeviceGetAttribute(&blocknum, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, 0);
    cudaDeviceGetAttribute(&sharedmemory, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&kernel, cudaDevAttrConcurrentKernels, 0);
    cudaDeviceGetAttribute(&overlap, cudaDevAttrGpuOverlap, 0);

    std::cout << "SM num\t\t\t" + std::to_string(SMnum) << std::endl;
    std::cout << "max block num per sm\t" + std::to_string(blocknum) << std::endl;
    std::cout << "max threads per blk\t" + std::to_string(threads) << std::endl;
    std::cout << "warp size\t\t" + std::to_string(warp) << std::endl;
    std::cout << "shared memory\t\t" + std::to_string(sharedmemory) << std::endl;
    std::cout << "concurrent kernels\t" + std::to_string(kernel) << std::endl;
    std::cout << "overlap\t\t\t" + std::to_string(overlap) << std::endl;

}

int main(int argc, char* argv[]) {
    // Print compile info
    compileInfo();

    // Print device properties
    printDeviceProp();

    // Create TaskManager
    TaskManager taskManager;

    // Create Task 1
    Task task1(1, "Task1");
    task1.addKernel(std::make_unique<OriCPKernel>(1, "Task1 CP 1"));
    task1.addKernel(std::make_unique<OriCPKernel>(2, "Task1 CP 2"));
    task1.addKernel(std::make_unique<OriCPKernel>(3, "Task1 CP 3"));

    Task task2(2, "Task2");
    task2.addKernel(std::make_unique<OriCPKernel>(1, "Task2 CP 1"));
    task2.addKernel(std::make_unique<OriCPKernel>(2, "Task2 CP 2"));
    task2.addKernel(std::make_unique<OriCPKernel>(3, "Task2 CP 3"));
    // Add tasks to the manager
    taskManager.addTask(task1);
    taskManager.addTask(task2);

    // Execute all tasks
    taskManager.executeAllTasks();

    return 0;
}
