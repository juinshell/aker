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
#include <unordered_map>
#include "./include/clipp.h"


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

#include "gptb_kernel/tzgemm_kernel.cu"
#include "tzgemm_kernel.h"

// tzgemm mix
#include "mix_kernel/tzgemm_cp.cu"
#include "mix_kernel/tzgemm_cutcp.cu"
#include "mix_kernel/tzgemm_fft.cu"
#include "mix_kernel/tzgemm_lbm.cu"
#include "mix_kernel/tzgemm_mrif.cu"
#include "mix_kernel/tzgemm_mriq.cu"
// #include "mix_kernel/tzgemm_sgemm.cu"
#include "mix_kernel/tzgemm_stencil.cu"

#include "json.h"
#include "Creator.h"

#ifndef SM_NUM
#define SM_NUM 68
#endif


Logger logger(LOG_FILE_PATH, true, true);

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
    {"gptb_tzgemm", (void*)general_ptb_tzgemm},
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
    {"cutcp_sgemm", (void*)mixed_cutcp_sgemm_kernel_1_1},
    {"tzgemm_cp", (void*)cp_tzgemm_mix},
    {"tzgemm_cutcp", (void*)cutcp_tzgemm_mix},
    {"tzgemm_fft", (void*)fft_tzgemm_mix},
    {"tzgemm_lbm", (void*)lbm_tzgemm_mix},
    {"tzgemm_mrif", (void*)mrif_tzgemm_mix},
    {"tzgemm_mriq", (void*)mriq_tzgemm_mix},
    {"tzgemm_stencil", (void*)stencil_tzgemm_mix},
};

void compileInfo() {
    std::cout << "Acker Version: " + std::to_string(Tacker_VERSION_MAJOR) + "." + std::to_string(Tacker_VERSION_MINOR) + "." + std::to_string(Tacker_VERSION_PATCH) << std::endl;
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

std::string SYSTEM = "aker";
std::string ROOT_PATH = "/home/jxdeng/workspace/tacker/runtime";

// extern std::unordered_map<std::string, GPTBKernel*> kernelMap;

int main(int argc, char* argv[]) {
    using namespace clipp;

    auto cli = (
        option("-s", "--sys", "--system").set(SYSTEM).doc("system name, one of aker/tacker/baymax"),
        option("-r", "--root").set(ROOT_PATH).doc("root path")
    );

    if(!parse(argc, argv, cli)) std::cout << make_man_page(cli, argv[0]);

    atexit (my_exit);

    read_json(ROOT_PATH + "/kinfo.json");

    initCUDA();
    // Print compile info
    compileInfo();

    // Print device properties
    printDeviceProp();
    system("nvidia-smi > nvidia-smi.log");

    // profile area
    // cudaEvent_t startKERNEL, stopKERNEL;
	// cudaErrCheck(cudaEventCreate(&startKERNEL));
	// cudaErrCheck(cudaEventCreate(&stopKERNEL));
    // float milliseconds = 0;
    // float ori_sum_time = 0;
    // float max_up = 0;
    // int cp_block_num = 32 * 512 / 10;
    // int m = 68, n = 1, k = 1;
    // int iter = 5;

    // printf("MAX_M_GLOBAL: %d, MAX_N_GLOBAL: %d, MAX_K_GLOBAL: %d\n", MAX_M_GLOBAL, MAX_N_GLOBAL, MAX_K_GLOBAL);

    // char foo;
    // cin>>foo;
    
    // while((m < MAX_M_GLOBAL) && (n < MAX_N_GLOBAL) && (k < MAX_K_GLOBAL)){
    //     auto o_cp = new OriCPKernel(0);
    //     auto g_cp = new GPTBKernel(
    //             10, 
    //             "cp",
    //             "gptb_cp", 
    //             o_cp, 
    //             dim3(SM_NUM * 8, 1, 1), 
    //             dim3(128, 1, 1), 
    //             0, 
    //             cp_block_num);
    //     auto o_tzgemm = new OriTZGEMMKernel(10, m, n, k);
    //     auto g_tzgemm = new GPTBKernel(
    //             11, 
    //             "tzgemm",
    //             "gptb_tzgemm", 
    //             o_tzgemm, 
    //             dim3(SM_NUM * 2, 1, 1), 
    //             dim3(128, 1, 1), 
    //             0, 
    //             getTZGEMMGridDim(m, n, k)[3]);
    //     auto mix_ = new MixKernel(
    //             100, 
    //             "tzgemm_cp", 
    //             g_cp, 
    //             g_tzgemm, 
    //             dim3(SM_NUM * 2, 1, 1), 
    //             dim3(256, 1, 1), 
    //             0, 
    //             cp_block_num,
    //             0,
    //             getTZGEMMGridDim(m, n, k)[3]);
        
    //     float ori1, ori2;
    //     cudaEventRecord(startKERNEL);
    //     o_cp->execute();
    //     cudaEventRecord(stopKERNEL);
    //     cudaEventSynchronize(stopKERNEL);
    //     cudaEventElapsedTime(&ori1, startKERNEL, stopKERNEL);

    //     cudaEventRecord(startKERNEL);
    //     o_tzgemm->execute();
    //     cudaEventRecord(stopKERNEL);
    //     cudaEventSynchronize(stopKERNEL);
    //     cudaEventElapsedTime(&ori2, startKERNEL, stopKERNEL);

    //     ori_sum_time = ori1 + ori2;
        
    //     cudaEventRecord(startKERNEL);
    //     mix_->execute();
    //     cudaEventRecord(stopKERNEL);
    //     cudaEventSynchronize(stopKERNEL);
    //     cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL);
    //     // std::cout << "block_ratio: " << cp_block_num * 1.0f / getTZGEMMGridDim(m, n, k)[3] << " mix_time: " << milliseconds << std::endl;


    //     if (max_up < (ori_sum_time - milliseconds) / ori_sum_time) {
    //         std::cout << "block_ratio: " << cp_block_num * 1.0f / getTZGEMMGridDim(m, n, k)[3] << std::endl;
    //         std::cout << "ori_sum_time: " << ori_sum_time << std::endl;
    //         std::cout << "mix_time: " << milliseconds << std::endl;
    //         max_up = (ori_sum_time - milliseconds) / ori_sum_time;
    //         std::cout << "max_up: " << (max_up * 1000.0f) << "%" << std::endl;
    //     }

    //     if (m + SM_NUM * iter < MAX_M_GLOBAL) m += SM_NUM * iter;
    //     else if (n + SM_NUM * iter < MAX_N_GLOBAL) n += SM_NUM * iter;
    //     else if (k + SM_NUM * iter < MAX_K_GLOBAL) k += SM_NUM * iter;
    //     else break;

    //     free(o_cp);
    //     free(g_cp);
    //     free(o_tzgemm);
    //     free(g_tzgemm);
    //     free(mix_);
    // }
    // auto x  = new MixKernel(
    //                 0, 
    //                 "cp_fft", 
    //                 createKernel("cp"),
    //                 createKernel("fft"),
    //                 dim3(SM_NUM * 2, 1, 1), 
    //                 dim3(1024, 1, 1), 
    //                 createKernel("cp")->gptbParams.ptb_start_block_pos,
    //                 createKernel("cp")->gptbParams.ptb_end_block_pos, 
    //                 createKernel("fft")->gptbParams.ptb_start_block_pos,
    //                 createKernel("fft")->gptbParams.ptb_end_block_pos);
    // x->execute();
    // cudaDeviceSynchronize();
    // sleep(1);

    // auto gptb_cp_ = createKernel("cp");
    // gptb_cp_->execute();

    // // Create Task 1
    // Task task1(1, "Task1");
    // task1.addKernel(std::make_unique<OriCPKernel>(1));
    // task1.addKernel(std::make_unique<OriCUTCPKernel>(2));
    // task1.addKernel(std::make_unique<OriFFTKernel>(3));
    // task1.addKernel(std::make_unique<OriLBMKernel>(4));
    // task1.addKernel(std::make_unique<OriMRIFKernel>(5));
    // task1.addKernel(std::make_unique<OriMRIQKernel>(6));
    // task1.addKernel(std::make_unique<OriSGEMMKernel>(7));
    // task1.addKernel(std::make_unique<OriSTENCILKernel>(8));
    // // task1.addKernel(std::make_unique<OriTZGEMMKernel>(9, 12544, 2048, 4608));

    // taskManager.addTask(task1);

    // Task task2(2, "task2");
    // auto oriCPKernel = std::make_unique<OriCPKernel>(10);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     10, 
    //     "gptb_cp", 
    //     std::move(oriCPKernel), 
    //     dim3(SM_NUM * 8, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     32 * 512));
    
    // auto oriCUTCPKernel = std::make_unique<OriCUTCPKernel>(11);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     11, 
    //     "gptb_cutcp", 
    //     std::move(oriCUTCPKernel), 
    //     dim3(SM_NUM * 6, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     1352));

    // auto oriFFTKernel = std::make_unique<OriFFTKernel>(12);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     12, 
    //     "gptb_fft", 
    //     std::move(oriFFTKernel), 
    //     dim3(SM_NUM * 3, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     10240));
    
    // auto oriLBMKernel = std::make_unique<OriLBMKernel>(16);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     16, 
    //     "gptb_lbm", 
    //     std::move(oriLBMKernel), 
    //     dim3(SM_NUM * 1, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     16384));

    // auto oriMRIFKernel = std::make_unique<OriMRIFKernel>(17);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     17, 
    //     "gptb_mrif", 
    //     std::move(oriMRIFKernel), 
    //     dim3(SM_NUM * 3, 1, 1), 
    //     dim3(256, 1, 1), 
    //     0, 
    //     1024));

    // auto oriMRIQKernel = std::make_unique<OriMRIQKernel>(18);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     18, 
    //     "gptb_mriq", 
    //     std::move(oriMRIQKernel), 
    //     dim3(SM_NUM * 4, 1, 1), 
    //     dim3(256, 1, 1), 
    //     0, 
    //     819));

    // auto oriSGEMMKernel = std::make_unique<OriSGEMMKernel>(19);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     19, 
    //     "gptb_sgemm", 
    //     std::move(oriSGEMMKernel), 
    //     dim3(SM_NUM * 4, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     774));

    // auto oriSTENCILKernel = std::make_unique<OriSTENCILKernel>(20);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     20, 
    //     "gptb_stencil", 
    //     std::move(oriSTENCILKernel), 
    //     dim3(SM_NUM * 3, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     1024));
    
    // auto oriTZGEMMKernel = std::make_unique<OriTZGEMMKernel>(21, 12544, 2048, 4608);
    // task2.addKernel(std::make_unique<GPTBKernel>(
    //     21,
    //     "gptb_tzgemm",
    //     std::move(oriTZGEMMKernel),
    //     dim3(SM_NUM * 1, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     oriTZGEMMKernel->launchGridDim.x));
    
    // taskManager.addTask(task2);

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

    // Resnet50 resnet50(3);

    // // taskManager.addTask(resnet50);

    // // test mix
    // auto tzgemm_cp_cp = std::make_unique<GPTBKernel>(
    //     0, 
    //     "gptb_cp", 
    //     std::make_unique<OriCPKernel>(0),
    //     dim3(SM_NUM * 6, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     32 * 512);

    // auto tzgemm_cp_tzgemm = std::make_unique<GPTBKernel>(
    //     1, 
    //     "gptb_tzgemm", 
    //     std::make_unique<OriTZGEMMKernel>(1, 4096, 2048, 2048),
    //     dim3(SM_NUM * 1, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     getTZGEMMGridDim(4096, 2048, 2048)[3]);
    
    // auto tzgemm_cp_mix = std::make_unique<MixKernel>(
    //     100, 
    //     "tzgemm_cp", 
    //     std::move(tzgemm_cp_cp), 
    //     std::move(tzgemm_cp_tzgemm), 
    //     dim3(SM_NUM * 1, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     32 * 512,
    //     0,
    //     getTZGEMMGridDim(4096, 2048, 2048)[3]);

    // Task task3(3, "task3");
    // task3.addKernel(std::make_unique<MixKernel>(
    //     100, 
    //     "tzgemm_cp", 
    //     std::move(tzgemm_cp_cp), 
    //     std::move(tzgemm_cp_tzgemm), 
    //     dim3(SM_NUM * 1, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0, 
    //     32 * 512,
    //     0,
    //     getTZGEMMGridDim(4096, 2048, 2048)[3]));

    // taskManager.addTask(task3);

    
    // for (int i = 0; i < 20; i++) {
    //     resnet50.executeTask(ExecutionMode::WARMUP);
    //     task1.executeTask(ExecutionMode::WARMUP);
    //     task2.executeTask(ExecutionMode::WARMUP);
    //     // task3.executeTask(ExecutionMode::WARMUP);
    // }

    // Execute all tasks
    auto lc_task = Resnet50(1000);

    TaskManager taskManager(&lc_task, "cp", "fft");

    // for (int i = 0; i < 5; i++) {
    //     taskManager.executeAllTasks(ExecutionMode::WARMUP);
    // }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    taskManager.executeAllTasks(ExecutionMode::PROFILE);

    // system("nvidia-smi >> nvidia-smi.log");

    return 0;
}
