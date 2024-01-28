// main.cc
#include "TackerConfig.h"
#include "util.h"
#include "TaskManager.h"
#include "Task.h"
#include "Kernel.h"
#include "Logger.h"
#include "ModuleCenter.h"
#include "Recorder.h"
#include <stdlib.h>


#include "cp_kernel.cu"
#include "cutcp_kernel.cu"
#include "fft_kernel.cu"
#include "lbm_kernel.cu"
#include "mrif_kernel.cu"
#include "mriq_kernel.cu"
#include "sgemm_kernel.cu"
#include "stencil_kernel.cu"

#include "GPTBKernel.h"
#include "MixKernel.h"

#include <unordered_map>
#include "gptb_kernel/cp_kernel.cu"
#include "gptb_kernel/cutcp_kernel.cu"
#include "gptb_kernel/fft_kernel.cu"
#include "gptb_kernel/lbm_kernel.cu"
#include "gptb_kernel/mrif_kernel.cu"
#include "gptb_kernel/mriq_kernel.cu"
#include "gptb_kernel/sgemm_kernel.cu"
#include "gptb_kernel/stencil_kernel.cu"

#include "mix_kernel/cp_fft_3_1.cu"
#include "mix_kernel/cp_sgemm_1_1.cu"
#include "mix_kernel/cutcp_fft_1_1.cu"
#include "mix_kernel/cutcp_sgemm_1_1.cu"
#include "mix_kernel/fft_lbm_6_1.cu"
#include "mix_kernel/fft_mriq_3_2.cu"
#include "mix_kernel/fft_sgemm_1_4.cu"
#include "mix_kernel/lbm_mrif_1_3.cu"
#include "mix_kernel/lbm_mriq_1_2.cu"
#include "mix_kernel/lbm_sgemm_1_7.cu"
#include "mix_kernel/mrif_sgemm_1_4.cu"
#include "mix_kernel/mriq_sgemm_1_2.cu"

// dnn
#include "dnn/resnet50/resnet50.h"




#ifndef SM_NUM
#define SM_NUM 68
#endif


Logger logger(LOG_FILE_PATH, true, true);

TaskManager taskManager;

ModuleCenter moduleCenter;

Recorder recorder;

std::unordered_map<std::string, void*> fmap = {
    {"gptb_cp", (void*)g_general_ptb_cp},
    {"gptb_cutcp", (void*)general_ptb_cutcp},
    {"gptb_fft", (void*)g_general_ptb_fft},
    {"gptb_lbm", (void*)general_ptb_lbm},
    {"gptb_mrif", (void*)g_general_ptb_mrif},
    {"gptb_mriq", (void*)g_general_ptb_mriq},
    {"gptb_sgemm", (void*)general_ptb_sgemm},
    {"gptb_stencil", (void*)general_ptb_stencil},
    {"cp_fft", (void*)mixed_cp_fft_kernel_3_1},
    {"cp_sgemm", (void*)mixed_cp_sgemm_kernel_1_1},
    {"fft_lbm", (void*)mixed_fft_lbm_kernel_6_1},
    {"fft_mriq", (void*)mixed_fft_mriq_kernel_3_2},
    {"fft_sgemm", (void*)mixed_fft_sgemm_kernel_1_4},
    {"lbm_mrif", (void*)mixed_lbm_mrif_kernel_1_3},
    {"lbm_mriq", (void*)mixed_lbm_mriq_kernel_1_2},
    {"lbm_sgemm", (void*)mixed_lbm_sgemm_kernel_1_7},
    {"mrif_sgemm", (void*)mixed_mrif_sgemm_kernel_1_4},
    {"mriq_sgemm", (void*)mixed_mriq_sgemm_kernel_1_2},
    {"cutcp_fft", (void*)mixed_cutcp_fft_kernel_1_1},
    {"cutcp_sgemm", (void*)mixed_cutcp_sgemm_kernel_1_1}
};

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

void my_exit() {
    recorder.text();
    system("nvidia-smi >> nvidia-smi.log");
    logger.INFO("Tacker exit");
}

int main(int argc, char* argv[]) {
    atexit (my_exit);
    initCUDA();
    // Print compile info
    compileInfo();

    // Print device properties
    printDeviceProp();
    system("nvidia-smi > nvidia-smi.log");

    // Create Task 1
    Task task1(1, "Task1");
    task1.addKernel(std::make_unique<OriCPKernel>(1));
    task1.addKernel(std::make_unique<OriCUTCPKernel>(2));
    task1.addKernel(std::make_unique<OriFFTKernel>(3));
    task1.addKernel(std::make_unique<OriLBMKernel>(4));
    task1.addKernel(std::make_unique<OriMRIFKernel>(5));
    task1.addKernel(std::make_unique<OriMRIQKernel>(6));
    task1.addKernel(std::make_unique<OriSGEMMKernel>(7));
    task1.addKernel(std::make_unique<OriSTENCILKernel>(8));

    taskManager.addTask(task1);

    Task task2(2, "task2");
    auto oriLBMKernel = std::make_unique<OriLBMKernel>(16);
    task2.addKernel(std::make_unique<GPTBKernel>(
        16, 
        "gptb_lbm", 
        std::move(oriLBMKernel), 
        dim3(SM_NUM * 1, 1, 1), 
        dim3(128, 1, 1), 
        0, 
        16384));

    auto oriMRIFKernel = std::make_unique<OriMRIFKernel>(17);
    task2.addKernel(std::make_unique<GPTBKernel>(
        17, 
        "gptb_mrif", 
        std::move(oriMRIFKernel), 
        dim3(SM_NUM * 3, 1, 1), 
        dim3(256, 1, 1), 
        0, 
        1024));

    auto oriMRIQKernel = std::make_unique<OriMRIQKernel>(18);
    task2.addKernel(std::make_unique<GPTBKernel>(
        18, 
        "gptb_mriq", 
        std::move(oriMRIQKernel), 
        dim3(SM_NUM * 4, 1, 1), 
        dim3(256, 1, 1), 
        0, 
        819));

    auto oriSGEMMKernel = std::make_unique<OriSGEMMKernel>(19);
    task2.addKernel(std::make_unique<GPTBKernel>(
        19, 
        "gptb_sgemm", 
        std::move(oriSGEMMKernel), 
        dim3(SM_NUM * 4, 1, 1), 
        dim3(128, 1, 1), 
        0, 
        774));

    auto oriSTENCILKernel = std::make_unique<OriSTENCILKernel>(20);
    task2.addKernel(std::make_unique<GPTBKernel>(
        20, 
        "gptb_stencil", 
        std::move(oriSTENCILKernel), 
        dim3(SM_NUM * 3, 1, 1), 
        dim3(128, 1, 1), 
        0, 
        1024));
    taskManager.addTask(task2);

    // Task task3(3, "Task3");
    // OriCPKernel oriCpKernel(3); // create lvalue
    // task3.addKernel(std::make_unique<GPTBKernel>(
    //     10, 
    //     "_Z16g_general_ptb_cpifPfiiiiiiiiii", 
    //     "ptb_cp", 
    //     static_cast<Kernel&>(oriCpKernel),
    //     dim3(SM_NUM * 6, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     32 * 512));
    
    // // bad
    // task3.addKernel(std::make_unique<GPTBKernel>( 
    //     11, 
    //     "_Z16g_general_ptb_cpifPfiiiiiiiiii", 
    //     "ptb_cp", 
    //     OriCPKernel(0, "gptb-make-cp"), // create rvalue
    //     dim3(SM_NUM * 6, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     32 * 512));
    
    // std::unique_ptr<OriCPKernel> oriCpKernel_ = std::make_unique<OriCPKernel>(5);
    // task3.addKernel(std::make_unique<GPTBKernel>(
    //     6, 
    //     "g_general_ptb_cp", 
    //     "cp_fft", 
    //     std::move(oriCpKernel_), // 不要使用unique_ptr，因为unique_ptr会在函数结束后被销毁，导致kernel的指针指向一个已经被销毁的对象
    //     dim3(SM_NUM * 6, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     32 * 512));

    Resnet50 resnet50(3);
    
    for (int i = 0; i < 10; i++) {
        resnet50.executeTask(ExecutionMode::WARMUP);
    }

    resnet50.executeTask(ExecutionMode::WARMUP);

    taskManager.addTask(resnet50);

    // Execute all tasks
    taskManager.executeAllTasks(ExecutionMode::PROFILE);

    system("nvidia-smi >> nvidia-smi.log");

    return 0;
}
