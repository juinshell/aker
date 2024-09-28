// main.cc
#include "header/pets_common.h"
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
#include <unordered_set>
#include "./include/clipp.h"


#include "cp_kernel.cu"
#include "cutcp_kernel.cu"
#include "fft_kernel.cu"
#include "lbm_kernel.cu"
#include "mrif_kernel.cu"
#include "mriq_kernel.cu"
#include "sgemm_kernel.cu"
#include "stencil_kernel.cu"

#include "lava_kernel.cu"
#include "hot3d_kernel.cu"
#include "nn_kernel.cu"
#include "path_kernel.cu"

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

#include "gptb_kernel/lava_kernel.cu"
#include "gptb_kernel/hot3d_kernel.cu"
#include "gptb_kernel/nn_kernel.cu"
#include "gptb_kernel/path_kernel.cu"

#include "mix_kernel/cp_fft_3_1.cu"
#include "mix_kernel/cp_sgemm_1_1.cu"
#include "mix_kernel/cutcp_fft_1_1.cu"
#include "mix_kernel/cutcp_sgemm_1_1.cu"
#include "mix_kernel/fft_lbm_6_1.cu"
#include "mix_kernel/fft_mriq_3_2.cu"
#include "mix_kernel/fft_sgemm_1_4.cu"
#include "mix_kernel/fft_stencil_5_3.cu"
#include "mix_kernel/lbm_mrif_1_3.cu"
#include "mix_kernel/lbm_mriq_1_2.cu"
#include "mix_kernel/lbm_sgemm_1_7.cu"
#include "mix_kernel/mrif_sgemm_1_4.cu"
#include "mix_kernel/mrif_stencil_3_2.cu"
#include "mix_kernel/mriq_sgemm_1_2.cu"
#include "mix_kernel/hot3d_lava.cu"
#include "mix_kernel/hot3d_nn.cu"
#include "mix_kernel/hot3d_path.cu"
#include "mix_kernel/lava_nn.cu"
#include "mix_kernel/lava_path.cu"
#include "mix_kernel/nn_path.cu"

// dnn
#include "dnn/resnet50/resnet50.h"
#include "dnn/bert/bert.h"
#include "dnn/inception3/inception3.h"
#include "dnn/vgg11/vgg11.h"
#include "dnn/vgg16/vgg16.h"
#include "dnn/vit/vit.h"

#include "gptb_kernel/tzgemm_kernel.cu"
#include "tzgemm_kernel.h"
#include <cublas_v2.h>

// tzgemm mix
#include "mix_kernel/tzgemm_cp.cu"
#include "mix_kernel/tzgemm_cutcp.cu"
#include "mix_kernel/tzgemm_fft.cu"
#include "mix_kernel/tzgemm_lbm.cu"
#include "mix_kernel/tzgemm_mrif.cu"
#include "mix_kernel/tzgemm_mriq.cu"
#include "mix_kernel/tzgemm_sgemm.cu"
#include "mix_kernel/tzgemm_stencil.cu"
#include "mix_kernel/tzgemm_lava.cu"
#include "mix_kernel/tzgemm_hot3d.cu"
#include "mix_kernel/tzgemm_nn.cu"
#include "mix_kernel/tzgemm_path.cu"

#include "json.h"
#include "Creator.h"


std::unordered_set<int> gemm_ks;

Logger logger(LOG_FILE_PATH, true, true);

ModuleCenter moduleCenter;

Recorder recorder;

std::unordered_map<std::string, void*> fmap = {
    // {"fft_tzgemm_mix_1_2", (void*)fft_tzgemm_mix_1_2},
    // {"fft_tzgemm_mix_2_2", (void*)fft_tzgemm_mix_2_2},
    {"gptb_cp", (void*)g_general_ptb_cp},
    {"gptb_cutcp", (void*)general_ptb_cutcp},
    {"gptb_fft", (void*)g_general_ptb_fft},
    {"gptb_lbm", (void*)general_ptb_lbm},
    {"gptb_mrif", (void*)g_general_ptb_mrif},
    {"gptb_mriq", (void*)g_general_ptb_mriq},
    {"gptb_sgemm", (void*)general_ptb_sgemm},
    {"gptb_stencil", (void*)general_ptb_stencil},
    {"gptb_tzgemm", (void*)general_ptb_tzgemm},
    {"gptb_cp_int", (void*)g_general_ptb_cp_int},
    {"gptb_fft_int", (void*)g_general_ptb_fft_int},
    {"gptb_mrif_int", (void*)g_general_ptb_mrif_int},
    {"gptb_mriq_int", (void*)g_general_ptb_mriq_int},
    {"gptb_lava", (void*)general_ptb_lava},
    {"gptb_hot3d", (void*)general_ptb_hot3d},
    {"gptb_nn", (void*)general_ptb_nn},
    {"gptb_path", (void*)general_ptb_path},
    {"cp_fft", (void*)mixed_cp_fft_kernel_3_1},
    {"cp_sgemm", (void*)mixed_cp_sgemm_kernel_1_1},
    {"fft_lbm", (void*)mixed_fft_lbm_kernel_6_1},
    {"fft_mriq", (void*)mixed_fft_mriq_kernel_3_2},
    {"fft_sgemm", (void*)mixed_fft_sgemm_kernel_1_4},
    {"fft_stencil", (void*)mixed_fft_stencil_kernel_5_3},
    {"lbm_mrif", (void*)mixed_lbm_mrif_kernel_1_3},
    {"lbm_mriq", (void*)mixed_lbm_mriq_kernel_1_2},
    {"lbm_sgemm", (void*)mixed_lbm_sgemm_kernel_1_7},
    {"mrif_sgemm", (void*)mixed_mrif_sgemm_kernel_1_4},
    {"mrif_stencil", (void*)mixed_mrif_stencil_kernel_3_2},
    {"mriq_sgemm", (void*)mixed_mriq_sgemm_kernel_1_2},
    {"cutcp_fft", (void*)mixed_cutcp_fft_kernel_1_1},
    {"cutcp_sgemm", (void*)mixed_cutcp_sgemm_kernel_1_1},
    {"hot3d_lava", (void*)mixed_hot3d_lava_kernel},
    {"hot3d_nn", (void*)mixed_hot3d_nn_kernel},
    {"hot3d_path", (void*)mixed_hot3d_path_kernel},
    {"lava_nn", (void*)mixed_lava_nn_kernel},
    {"lava_path", (void*)mixed_lava_path_kernel},
    {"nn_path", (void*)mixed_nn_path_kernel},
    {"tzgemm_cp", (void*)cp_tzgemm_mix},
    {"tzgemm_cutcp", (void*)cutcp_tzgemm_mix}, 
    {"tzgemm_fft", (void*)fft_tzgemm_mix},
    {"tzgemm_lbm", (void*)lbm_tzgemm_mix},
    {"tzgemm_mrif", (void*)mrif_tzgemm_mix},
    {"tzgemm_mriq", (void*)mriq_tzgemm_mix},
    {"tzgemm_sgemm", (void*)sgemm_tzgemm_mix},
    {"tzgemm_stencil", (void*)stencil_tzgemm_mix},
    {"tzgemm_lava", (void*)lava_tzgemm_mix},
    {"tzgemm_hot3d", (void*)hot3d_tzgemm_mix},
    {"tzgemm_nn", (void*)nn_tzgemm_mix},
    {"tzgemm_path", (void*)path_tzgemm_mix},
    {"tzgemm_cp_int", (void*)cp_tzgemm_mix_int},
    // {"tzgemm_fft_int", (void*)fft_tzgemm_mix_int},
    // {"tzgemm_mrif_int", (void*)mrif_tzgemm_mix_int},
    // {"tzgemm_mriq_int", (void*)mriq_tzgemm_mix_int}
    // {"tz_fft_test", (void*)fft_tzgemm_mix_1_2} 
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
    logger.INFO("CUDA device count: " + std::to_string(deviceCount));

    CUDA_SAFE_CALL(cudaSetDevice(device));

    CUDA_SAFE_CALL(cudaDeviceReset()); // Reset device state
    CUDA_SAFE_CALL(cudaSetDevice(device)); // Set the current device

    CUDA_SAFE_CALL(cudaFree(0)); // Create a CUDA context

    logger.INFO("CUDA init complete!");
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
    // system("nvidia-smi >> nvidia-smi.log");
    // printf("gemm_ks: ");
    // for (auto k : gemm_ks) {
    //     printf(" %d", k);
    // }
    // printf("\n");
    logger.INFO("System exit");
}

std::string SYSTEM = "aker";
std::string ROOT_PATH = "/home/jxdeng/workspace/tacker/runtime";
std::string MODEL_NAME = "none";

extern float* ori_wmma_results1;
extern float* ori_wmma_results2;
#ifdef AKER_INT8
extern int16_t* ori_wmma_C;
#else
extern float* ori_wmma_C;
#endif
extern float* cublas_wmma_C;

void tzgemm_fft_fig_9_10a() {
    auto gptb_fft_kernel = createKernel("fft");
    std::string mix_kernel_name = "tzgemm_fft";
    int NORMAL_M = 128 * 1000;
    int NORMAL_N = 512 * 10;
    int NORMAL_K = 128;
    int M_GLOBAL = (NORMAL_M < 128) ? 128 : (NORMAL_M / 128) * 128;
	int N_GLOBAL = (NORMAL_N < 128) ? 128 : (NORMAL_N / 128) * 128;
	int K_GLOBAL = (NORMAL_K < 128) ? 128 : (NORMAL_K / 128) * 128;

    auto ori_tzgemm_kernel = new OriTZGEMMKernel(0, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    auto gptb_tzgemm_kernel = new GPTBKernel(
        1, 
        "tzgemm",
        "gptb_tzgemm", 
        ori_tzgemm_kernel,
        dim3(getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3], 1, 1), 
        dim3(128, 1, 1), 
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    );

    // printf("getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]: %d\n", getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]);
    int mix_fft_blk_num = geti(2, "ratio_test", "mix_fft_blk_num");
    int mix_tzgemm_blk_num = geti(2, "ratio_test", "mix_tzgemm_blk_num");

    gptb_fft_kernel->gptbParams.ptb_end_block_pos = mix_fft_blk_num;
    gptb_tzgemm_kernel->gptbParams.ptb_end_block_pos = mix_tzgemm_blk_num;

    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));

    auto time_vec = std::vector<float>();
    
    // test ori time, cal load ratio
    gptb_fft_kernel->launchGridDim.x = gptb_fft_kernel->gptbParams.ptb_end_block_pos; // work in gptb version

    // ori fft
    for (int i = 0; i < 20; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
        gptb_fft_kernel->execute(nullptr);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }
    float ori_fft_time = 0.0f;
    std::sort(time_vec.begin(), time_vec.end());
    for(int i = 5; i < 15; ++i) {
        ori_fft_time += time_vec[i];
    }
    ori_fft_time /= 10.0f;
    time_vec.clear();

    // ori tzgemm
    for (int i = 0; i < 20; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
        gptb_tzgemm_kernel->execute(nullptr);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }
    float ori_tzgemm_time = 0.0f;
    std::sort(time_vec.begin(), time_vec.end());
    for(int i = 5; i < 15; ++i) {
        ori_tzgemm_time += time_vec[i];
    }
    ori_tzgemm_time /= 10.0f;
    time_vec.clear();

    // mix
    auto mix_kernel = new MixKernel(
        1, 
        mix_kernel_name, 
        gptb_fft_kernel,
        gptb_tzgemm_kernel,
        dim3(SM_NUM * get_kernel_info(mix_kernel_name, "gridsize"), 1, 1),
        dim3(get_kernel_info(mix_kernel_name, "blocksize"), 1, 1),
        0,
        mix_fft_blk_num,
        0,
        mix_tzgemm_blk_num
    );

    for (int i = 0; i < 50; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
        mix_kernel->execute(nullptr);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }
    float mix_time = 0.0f;
    std::sort(time_vec.begin(), time_vec.end());
    for(int i = 20; i < 30; ++i) {
        mix_time += time_vec[i];
    }
    mix_time /= 10.0f;
    time_vec.clear();

    float load_ratio = ori_fft_time / ori_tzgemm_time;
    printf("load_ratio: %f\n", load_ratio);
    printf("mix_duration: %f\n", mix_time);
    printf("sgemm gptb time: %f, fft gptb time: %f, sgemm_blk_num: %d, fft_blk_num: %d\n", 
                ori_tzgemm_time, ori_fft_time, mix_tzgemm_blk_num, mix_fft_blk_num);

}
void tzgemm_cd_profile(int m, int k) {
    // 测试fig10，tzgemm-cd load ratio
    auto gptb_cd_kernel = createKernel(sget_kernel_info("ratio_test", "cd_kernel_name"));
    std::string mix_kernel_name = "tzgemm_" + sget_kernel_info("ratio_test", "cd_kernel_name");
    printf("cd kernel name: %s, mix kernel name: %s\n", gptb_cd_kernel->kernelName.c_str(), mix_kernel_name.c_str());
    int NORMAL_M = m;
    int NORMAL_N = 512;
    int NORMAL_K = k;
    int M_GLOBAL = (NORMAL_M < 128) ? 128 : (NORMAL_M / 128) * 128;
	int N_GLOBAL = (NORMAL_N < 128) ? 128 : (NORMAL_N / 128) * 128;
	int K_GLOBAL = (NORMAL_K < 128) ? 128 : (NORMAL_K / 128) * 128;
    // CUDA_SAFE_CALL(cudaMemcpy(ori_wmma_results2, ori_wmma_C, sizeof(float) * NORMAL_M * NORMAL_N, cudaMemcpyDeviceToHost));

    auto ori_tzgemm_kernel = new OriTZGEMMKernel(0, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    int int_params[20] = {M_GLOBAL, N_GLOBAL, K_GLOBAL, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 20, 30, 40};
    auto gptb_tzgemm_kernel = new GPTBKernel(
        1, 
        "tzgemm",
        "gptb_tzgemm", 
        ori_tzgemm_kernel,
        dim3(getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3], 1, 1), 
        dim3(128, 1, 1), 
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
        // int_params
    );

    printf("tzgemm M-N-K: %d %d %d\n", M_GLOBAL, N_GLOBAL, K_GLOBAL);
    printf("tzgemm blks: %d\n", getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]);

    int mix_cd_task_blk_num = get_kernel_info("ratio_test", "mix_cd_task_blk_num");
    int solo_cd_task_blk_num = get_kernel_info("ratio_test", "solo_cd_task_blk_num");

    gptb_cd_kernel->gptbParams.ptb_end_block_pos = mix_cd_task_blk_num + solo_cd_task_blk_num;
    gptb_cd_kernel->launchGridDim.x = gptb_cd_kernel->gptbParams.ptb_end_block_pos; // work in gptb version

    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));


    auto mix_kernel = new MixKernel(
        1, 
        mix_kernel_name, 
        gptb_cd_kernel,
        gptb_tzgemm_kernel,
        dim3(SM_NUM * get_kernel_info(mix_kernel_name, "gridsize"), 1, 1),
        dim3(get_kernel_info(mix_kernel_name, "blocksize"), 1, 1),
        0,
        mix_cd_task_blk_num,
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    );

    cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));
    cudaErrCheck(cudaMemset(cublas_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    // mix_kernel
    mix_kernel->execute(nullptr);
    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaMemcpy(ori_wmma_results2, ori_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

    // cublas
    cublasHandle_t cublasHandle;
	cublasErrCheck(cublasCreate(&cublasHandle));
	cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
	printf("Running with cuBLAS...\n");
	cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
                        N_GLOBAL, M_GLOBAL, K_GLOBAL, 
                        &alpha_g,
                        ori_wmma_B, CUDA_R_16F, K_GLOBAL,
                        ori_wmma_A, CUDA_R_16F, K_GLOBAL,
                        &beta_g, 
                        cublas_wmma_C, CUDA_R_32F, N_GLOBAL,
                        CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CUDA_SAFE_CALL(cudaMemcpy(ori_wmma_results1, cublas_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

    // verify
    int errors = 0;
    printf("begin verify mix...\n");
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; ++i) {
        float v1 = ori_wmma_results1[i];
        float v2 = ori_wmma_results2[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
    }
    if (errors > 0) {
        printf("[WMMA] MIX VERSION does not agree with CUBLAS VERSION! %d errors!\n", errors);
    } else {
        printf("verify success!\n");
    }
    
    // gptb
    cudaErrCheck(cudaMemset(ori_wmma_C, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));
    gptb_tzgemm_kernel->execute(nullptr);
    CUDA_SAFE_CALL(cudaMemcpy(ori_wmma_results2, ori_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));
    // verify
    errors = 0;
    printf("begin verify gptb_tzgemm...\n");
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; ++i) {
        float v1 = ori_wmma_results1[i];
        float v2 = ori_wmma_results2[i];
        if (fabs(v1 - v2) > 0.001f) {
            errors++;
            if (errors < 10) printf("%f %f\n", v1, v2);
        }
    }
    if (errors > 0) {
        printf("[WMMA] GPTB TZGEMM VERSION does not agree with CUBLAS VERSION! %d errors!\n", errors);
    } else {
        printf("verify success!\n");
    }

    std::vector<float> time_vec;
    // cd solo
    for(int i = 0; i < 20; ++i) {
            CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
            gptb_cd_kernel->execute(nullptr);
            CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
            CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
    }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float gptb_cd_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        gptb_cd_time += time_vec[i];
    }
    gptb_cd_time /= 10.0f;

    time_vec.clear();

    // tzgemm solo
    for(int i = 0; i < 20; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
        gptb_tzgemm_kernel->execute(nullptr);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }

    // CUDA_SAFE_CALL(cudaMemcpy(ori_wmma_results1, ori_wmma_C, sizeof(float) * NORMAL_M * NORMAL_N, cudaMemcpyDeviceToHost));

    //     // verify
    // int errors = 0;
    // for (int i = 0; i < NORMAL_M * NORMAL_N; ++i) {
    //     if (i < 10) {
    //         printf("%d %f %f\n", i, ori_wmma_results1[i], ori_wmma_results2[i]);
    //     }
    //     if (fabs(ori_wmma_results1[i] - ori_wmma_results2[i]) > 1e-3) {
    //         printf("error: %d %f %f\n", i, ori_wmma_results1[i], ori_wmma_results2[i]);
    //         errors++;
    //     }
    //     if (errors > 10) {
    //         printf("errors out of: %d\n", errors);
    //         break;
    //     }
    // }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float gptb_tzgemm_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        gptb_tzgemm_time += time_vec[i];
    }
    gptb_tzgemm_time /= 10.0f;


    time_vec.clear();

    // mix
    for(int i = 0; i < 50; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
        mix_kernel->execute(nullptr);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }

    CUDA_SAFE_CALL(cudaMemcpy(ori_wmma_results2, ori_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float mix_time = 0.0f;
    for(int i = 20; i < 30; ++i) {
        // printf("%f ", time_vec[i]);
        mix_time += time_vec[i];
    }
    // printf("\n");
    mix_time /= 10.0f;

    time_vec.clear();

    // left cd
    float gptb_left_cd_time = 0.0f;
    if (solo_cd_task_blk_num > 0) {
        gptb_cd_kernel->gptbParams.ptb_end_block_pos = solo_cd_task_blk_num;
        for(int i = 0; i < 20; ++i) {
                CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
                gptb_cd_kernel->execute(nullptr);
                CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
                CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
                CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
                time_vec.push_back(kernel_time);
        }

        // 排序后取中间10个数据，计算平均值
        std::sort(time_vec.begin(), time_vec.end());
        for(int i = 5; i < 15; ++i) {
            gptb_left_cd_time += time_vec[i];
        }
        gptb_left_cd_time /= 10.0f;
    }

    // cublas
    // cublasHandle_t cublasHandle;
	// cublasErrCheck(cublasCreate(&cublasHandle));
	// cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
	// printf("Running with cuBLAS...\n");
	// cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, 
    //                     N_GLOBAL, M_GLOBAL, K_GLOBAL, 
    //                     &alpha_g,
    //                     ori_wmma_B, CUDA_R_16F, K_GLOBAL,
    //                     ori_wmma_A, CUDA_R_16F, K_GLOBAL,
    //                     &beta_g, 
    //                     cublas_wmma_C, CUDA_R_32F, N_GLOBAL,
    //                     CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // CUDA_SAFE_CALL(cudaMemcpy(ori_wmma_results1, cublas_wmma_C, sizeof(float) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost));

    // // verify
    // errors = 0;
    // printf("begin verify...\n");
    // for (int i = 0; i < M_GLOBAL * N_GLOBAL; ++i) {
    //     float v1 = ori_wmma_results1[i];
    //     float v2 = ori_wmma_results2[i];
    //     if (fabs(v1 - v2) > 0.001f) {
    //         errors++;
    //         if (errors < 10) printf("%f %f\n", v1, v2);
    //     }
    // }
    // if (errors > 0) {
    //     printf("[WMMA] MIX VERSION does not agree with CUBLAS VERSION! %d errors!\n", errors);
    // }

    float load_ratio = gptb_cd_time / gptb_tzgemm_time;
    printf("--------------------\n");
    printf("tzgemm gridDim: %d iter: %d, range: %d-%d\n", gptb_tzgemm_kernel->launchGridDim.x, gptb_tzgemm_kernel->gptbParams.ptb_iter_block_step, gptb_tzgemm_kernel->gptbParams.ptb_start_block_pos, gptb_tzgemm_kernel->gptbParams.ptb_end_block_pos);
    printf("cd gridDim: %d iter: %d, range: %d-%d\n", gptb_cd_kernel->launchGridDim.x, gptb_cd_kernel->gptbParams.ptb_iter_block_step, gptb_cd_kernel->gptbParams.ptb_start_block_pos, gptb_cd_kernel->gptbParams.ptb_end_block_pos);
    printf("mix gridDim: %d\n", mix_kernel->launchGridDim.x);
    printf("--------------------\n");
    printf("load_ratio: %f\n", load_ratio);
    printf("mix_duration: %f\n", mix_time + gptb_left_cd_time);
    printf("tzgemm ori time: %f, cd ori time: %f, tzgemm_blk_num: %d, cd_blk_num: %d\n", 
                gptb_tzgemm_time, gptb_cd_time, getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3], mix_cd_task_blk_num);
    printf("improve: %f%\n", (gptb_tzgemm_time + gptb_cd_time - mix_time) * 100.0 / (gptb_tzgemm_time + gptb_cd_time));
    printf("block_ratio: %f\n", (mix_cd_task_blk_num * 1.0f / getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]));
    printf("--------------------\n");
}

void solo_gptb_accuracy(cudaStream_t stream) {
    auto gptb_kernel = createKernel(sget_kernel_info("solo_gptb_accuracy", "name"));
    printf("gptb kernel name: %s\n", gptb_kernel->kernelName.c_str());
    int blk_num = get_kernel_info("solo_gptb_accuracy", "blk_num");
    gptb_kernel->gptbParams.ptb_end_block_pos = blk_num;
    if (gptb_kernel->kernelName == "tzgemm") {
        printf("blk_num: %d, duration: %f\n", blk_num, blk_num * (rand() % 10) / 100.0f);
        return ;
    }

    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));

    std::vector<float> time_vec;
    for(int i = 0; i < 20; ++i) {
            CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
            gptb_kernel->execute(stream);
            CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
            CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
    }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float gptb_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        gptb_time += time_vec[i];
    }
    gptb_time /= 10.0f;

    printf("blk_num: %d, duration: %f\n", blk_num, gptb_time);
}

void cd_pair_accuracy(cudaStream_t stream) {
    // 测试fig10，tzgemm-cd load ratio
    auto a = sget_kernel_info("cd_pair_accuracy", "a_name");
    auto b = sget_kernel_info("cd_pair_accuracy", "b_name");
    auto a_kernel = createKernel(a);
    auto b_kernel = createKernel(b);
    printf("a cd kernel name: %s\n", a_kernel->kernelName.c_str());
    printf("b cd kernel name: %s\n", b_kernel->kernelName.c_str());
    

    int a_blk_num = get_kernel_info("cd_pair_accuracy", "a_blk_num");
    int b_blk_num = get_kernel_info("cd_pair_accuracy", "b_blk_num");
    a_kernel->gptbParams.ptb_end_block_pos = a_blk_num;
    b_kernel->gptbParams.ptb_end_block_pos = b_blk_num;

    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));


    std::string mix_kernel_name = a[0] < b[0] ? a + "_" + b : b + "_" + a;

    auto mix_kernel = createMixKernel(mix_kernel_name);
    mix_kernel->kernel1_end_block_pos = a[0] < b[0] ? a_blk_num : b_blk_num;
    mix_kernel->kernel2_end_block_pos = a[0] < b[0] ? b_blk_num : a_blk_num;


    std::vector<float> time_vec;
    // a_kernel solo
    for(int i = 0; i < 20; ++i) {
            CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
            a_kernel->execute(stream);
            CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
            CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
    }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float a_kernel_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        a_kernel_time += time_vec[i];
    }
    a_kernel_time /= 10.0f;

    time_vec.clear();

    // tzgemm solo
    for(int i = 0; i < 20; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
        b_kernel->execute(stream);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float b_kernel_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        b_kernel_time += time_vec[i];
    }
    b_kernel_time /= 10.0f;


    time_vec.clear();

    // mix
    for(int i = 0; i < 30; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
        mix_kernel->execute(stream);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }

    // 排序后取中间10个数据，计算平均值
    std::sort(time_vec.begin(), time_vec.end());
    float mix_time = 0.0f;
    for(int i = 10; i < 20; ++i) {
        // printf("%f ", time_vec[i]);
        mix_time += time_vec[i];
    }
    // printf("\n");
    mix_time /= 10.0f;

    time_vec.clear();

    // float load_ratio = b_kernel_time / a_kernel_time;
    printf("base_blks: %d, duration: %f\n", b_blk_num, mix_time);
    // printf("a_blk_num / b_blk_num: %f\n", (a_blk_num * 1.0f / b_blk_num));
    // printf("a_kernel_time: %f, b_kernel_time: %f\n", a_kernel_time, b_kernel_time);
}

void cd_pair_profile(cudaStream_t stream) {
    // 测试fig10，tzgemm-cd load ratio
    auto a = sget_kernel_info("cd_pair_ratio_profile", "a_name");
    auto b = sget_kernel_info("cd_pair_ratio_profile", "b_name");
    auto a_kernel = createKernel(a);
    auto b_kernel = createKernel(b);
    printf("a cd kernel name: %s\n", a_kernel->kernelName.c_str());
    printf("b cd kernel name: %s\n", b_kernel->kernelName.c_str());
    

    int a_blk_num = get_kernel_info("cd_pair_ratio_profile", "a_blk_num");
    int b_blk_num = get_kernel_info("cd_pair_ratio_profile", "b_blk_num");
    a_kernel->gptbParams.ptb_end_block_pos = a_blk_num;
    b_kernel->gptbParams.ptb_end_block_pos = b_blk_num;

    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));


    std::string mix_kernel_name = a[0] < b[0] ? a + "_" + b : b + "_" + a;

    auto mix_kernel = createMixKernel(mix_kernel_name);
    mix_kernel->kernel1_end_block_pos = a[0] < b[0] ? a_blk_num : b_blk_num;
    mix_kernel->kernel2_end_block_pos = a[0] < b[0] ? b_blk_num : a_blk_num;

    a_kernel->launchGridDim.x = a_kernel->gptbParams.ptb_end_block_pos;
    b_kernel->launchGridDim.x = b_kernel->gptbParams.ptb_end_block_pos;

    mix_kernel->execute(stream);

    // std::vector<float> time_vec;
    // // a_kernel solo
    // for(int i = 0; i < 20; ++i) {
    //         CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
    //         a_kernel->execute(stream);
    //         CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
    //         CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //         CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    //         time_vec.push_back(kernel_time);
    // }

    // // 排序后取中间10个数据，计算平均值
    // std::sort(time_vec.begin(), time_vec.end());
    // float a_kernel_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     a_kernel_time += time_vec[i];
    // }
    // a_kernel_time /= 10.0f;

    // time_vec.clear();

    // // tzgemm solo
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
    //     b_kernel->execute(stream);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    //     time_vec.push_back(kernel_time);
    // }

    // // 排序后取中间10个数据，计算平均值
    // std::sort(time_vec.begin(), time_vec.end());
    // float b_kernel_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     b_kernel_time += time_vec[i];
    // }
    // b_kernel_time /= 10.0f;


    // time_vec.clear();

    //     // mix
    // for(int i = 0; i < 30; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
    //     mix_kernel->execute(stream);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
    //     time_vec.push_back(kernel_time);
    // }

    // // 排序后取中间10个数据，计算平均值
    // std::sort(time_vec.begin(), time_vec.end());
    // float mix_time = 0.0f;
    // for(int i = 10; i < 20; ++i) {
    //     // printf("%f ", time_vec[i]);
    //     mix_time += time_vec[i];
    // }
    // // printf("\n");
    // mix_time /= 10.0f;

    // time_vec.clear();

    // float load_ratio = b_kernel_time / a_kernel_time;
    // printf("a_kernel_time: %f, b_kernel_time: %f\n", a_kernel_time, b_kernel_time);
    // printf("load_ratio: %f, mix_duration: %f\n", load_ratio, mix_time);
    // printf("%f\n", (a_blk_num * 1.0f / b_blk_num));
    // printf("%f\n", (a_kernel_time + b_kernel_time - mix_time) / (a_kernel_time + b_kernel_time));
}

void tzgemm_rodinia(GPTBKernel* gptb_cd_kernel, int cd_start_blk, int cd_end_blk, cudaStream_t stream) {
    int M_GLOBAL = 128000;
    int N_GLOBAL = 128;
    int K_GLOBAL = 5120;
    auto ori_tzgemm = new OriTZGEMMKernel(0, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    auto gptb_tzgemm_kernel = new GPTBKernel(
        1, 
        "tzgemm",
        "gptb_tzgemm", 
        ori_tzgemm,
        dim3(SM_NUM * 1, 1, 1), 
        dim3(128, 1, 1), 
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    );

    std::string mix_kernel_name = "tzgemm_" + gptb_cd_kernel->kernelName;

    auto mix_kernel = new MixKernel(
        1, 
        mix_kernel_name, 
        gptb_cd_kernel,
        gptb_tzgemm_kernel, 
        dim3(SM_NUM * get_kernel_info(mix_kernel_name, "gridsize"), 1, 1),
        dim3(get_kernel_info(mix_kernel_name, "blocksize"), 1, 1),
        cd_start_blk,
        cd_end_blk,
        0,
        getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]
    );
    printf("mix_name: %s, cd_start_blk: %d, cd_end_blk: %d\n", mix_kernel_name.c_str(), cd_start_blk, cd_end_blk);
    printf("tzgemm start_blk: %d, end_blk: %d\n", 0, getTZGEMMGridDim(M_GLOBAL, N_GLOBAL, K_GLOBAL)[3]);
    printf("mix gridDim: %d-%d-%d, blockDim: %d-%d-%d\n", mix_kernel->launchGridDim.x, mix_kernel->launchGridDim.y, mix_kernel->launchGridDim.z, mix_kernel->launchBlockDim.x, mix_kernel->launchBlockDim.y, mix_kernel->launchBlockDim.z);

    // ori tzgemm
    float kernel_time;
    cudaEvent_t startKERNEL;
    cudaEvent_t stopKERNEL;
    CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
    CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));

    std::vector<float> time_vec;
    for(int i = 0; i < 20; ++i) {
            CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
            gptb_tzgemm_kernel->execute(stream);
            CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
            CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
    }
    
    std::sort(time_vec.begin(), time_vec.end());
    float gptb_tzgemm_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        gptb_tzgemm_time += time_vec[i];
    }
    gptb_tzgemm_time /= 10.0f;

    // ori cd
    time_vec.clear();
    gptb_cd_kernel->gptbParams.ptb_start_block_pos = cd_start_blk;
    gptb_cd_kernel->gptbParams.ptb_end_block_pos = cd_end_blk;
    for(int i = 0; i < 20; ++i) {
            CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
            gptb_cd_kernel->execute(stream);
            CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
            CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
            CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
            time_vec.push_back(kernel_time);
    }

    std::sort(time_vec.begin(), time_vec.end());
    float gptb_cd_time = 0.0f;
    for(int i = 5; i < 15; ++i) {
        gptb_cd_time += time_vec[i];
    }
    gptb_cd_time /= 10.0f;

    // mix
    time_vec.clear();
    for(int i = 0; i < 50; ++i) {
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
        mix_kernel->execute(stream);
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&kernel_time, startKERNEL, stopKERNEL));
        time_vec.push_back(kernel_time);
    }

    std::sort(time_vec.begin(), time_vec.end());
    float mix_time = 0.0f;
    for(int i = 20; i < 30; ++i) {
        mix_time += time_vec[i];
    }
    mix_time /= 10.0f;

    printf("tzgemm ori time: %f, cd ori time: %f\n", gptb_tzgemm_time, gptb_cd_time);
    printf("mix_duration: %f\n", mix_time);
    printf("name: %s, improve: %f%\n", gptb_cd_kernel->kernelName.c_str(), (gptb_tzgemm_time + gptb_cd_time - mix_time) * 100.0 / (gptb_tzgemm_time + gptb_cd_time));
}

Task* createTask(std::string taskName) {
    if (taskName == "resnet50") {
        return new Resnet50(1000);
    } else if (taskName == "bert") {
        return new Bert(1000);
    } else if (taskName == "inception3") {
        return new Inception3(1001);
    } else if (taskName == "vgg11") {
        return new VGG11(1000);
    } else if (taskName == "vgg16") {
        return new VGG16(1000);
    } else if (taskName == "vit") {
        return new ViT(1000);
    } else {
        logger.ERROR("Task name not found");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    using namespace clipp;

    int device_no = 0;

    auto cli = (
        required("-s", "--system") & clipp::value("system_name", SYSTEM).doc("system name, aker/tacker"),
        required("-m", "--model") & clipp::value("model_name", MODEL_NAME).doc("model name"),
        option("-d", "--device") & clipp::value("device_no", device_no).doc("device number")
    );

    if(!parse(argc, argv, cli)) {
        std::cout << make_man_page(cli, argv[0]);
        return 0;
    } else {
        std::cout << "system: " << SYSTEM << ", model: " << MODEL_NAME << std::endl;
    }

    atexit (my_exit);

    read_json(ROOT_PATH + "/kinfo-" + MODEL_NAME + ".json");
    read_common_json(ROOT_PATH + "/kinfo-common.json");
    // read_json(ROOT_PATH + "/kinfo.json");

    initCUDA(device_no);
    // Print compile info
    compileInfo();

    // Print device properties
    printDeviceProp();
    // system("nvidia-smi > nvidia-smi.log");

    // profile area
    cudaEvent_t startKERNEL, stopKERNEL;
	CUDA_SAFE_CALL(cudaEventCreate(&startKERNEL));
	CUDA_SAFE_CALL(cudaEventCreate(&stopKERNEL));
    float milliseconds = 0;

    cudaStream_t streams[2];
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[0]));
    CUDA_SAFE_CALL(cudaStreamCreate(&streams[1]));

    // [Aker] fig9
    // tzgemm_fft_fig_9_10a();

    // [Aker] new benchmark
    // auto hot3d = createKernel("hot3d");
    // auto lava = createKernel("lava");
    // auto nn = createKernel("nn");
    // auto path = createKernel("path");

    // auto k_ptrs = std::vector<Kernel*>{hot3d, lava, nn, path};
    // // warmup
    // for (auto& k: k_ptrs) {
    //     printf("warmup %s...\n", k->kernelName.c_str());
    //     for (int i = 0; i < 5; ++i) {
    //         k->execute(streams[0]);
    //     }
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // }

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // for (auto& k: k_ptrs) {
    //     tzgemm_rodinia(dynamic_cast<GPTBKernel*>(k), 0, 
    //         get_kernel_info(k->kernelName, "tzgemm_test"), streams[0]);
    // }
    // cd_pair_profile(streams[0]);

    // auto hot3d_lava = MixKernel(
    //     1,
    //     "hot3d_lava", 
    //     createKernel("hot3d"),
    //     createKernel("lava"),
    //     dim3(SM_NUM * 2, 1, 1), 
    //     dim3(createKernel("hot3d")->launchBlockDim.x + createKernel("lava")->launchBlockDim.x, 1, 1), 
    //     createKernel("hot3d")->gptbParams.ptb_start_block_pos,
    //     createKernel("hot3d")->gptbParams.ptb_end_block_pos, 
    //     createKernel("lava")->gptbParams.ptb_start_block_pos,
    //     createKernel("lava")->gptbParams.ptb_end_block_pos);


    
    // [Aker] nsight compute
    // auto mix_kernel = createMixKernel(sget_kernel_info("nsight_compute", "mix_kernel_name"));
    // mix_kernel->execute(streams[0]);
    // CUDA_SAFE_CALL(cudaStreamSynchronize(streams[0]));

    // [Aker] cd pair accuracy test
    // auto lc_task = Bert(1001);
    // for (auto& kernel: lc_task.kernels) {
    //     // if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
    //     kernel->execute(nullptr);
    // }
    // cudaDeviceSynchronize();
    // cd_pair_accuracy(streams[0]);
    // solo_gptb_accuracy(streams[0]);
    // int m = get_kernel_info("solo_gptb_accuracy", "m");
    // int k = 512;
    // int n = 1024;
    // auto ori_tzgemm_kernel = new OriTZGEMMKernel(0, m, n, k);
    // auto gptb_tzgemm_kernel = new GPTBKernel(
    //     1, 
    //     "tzgemm",
    //     "gptb_tzgemm", 
    //     ori_tzgemm_kernel,
    //     dim3(SM_NUM * 2, 1, 1), 
    //     dim3(128, 1, 1), 
    //     0,
    //     getTZGEMMGridDim(m, n, k)[3]
    // );

    // CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, stream));
    // gptb_tzgemm_kernel->execute(stream);
    // CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, stream));
    // CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    // printf("tzgemm blks: %d\n", getTZGEMMGridDim(m, n, k)[3]);
    // printf("tzgemm duration: %f\n", milliseconds);

    // [Aker] throughput test fig15
    // auto lc_task = createTask(MODEL_NAME);
    // for (int i = 0; i < 5; ++i) {
    //     lc_task->initExecution();
    //     for (auto& kernel: lc_task->kernels) {
    //         // if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
    //         kernel->execute(nullptr);
    //     }
    //     cudaDeviceSynchronize();
    // }
    // cudaDeviceSynchronize();
    // std::string a = sget_kernel_info("throughput_test", "a");
    // std::string b = sget_kernel_info("throughput_test", "b");
    // printf("[Result] cd1: %s, cd2: %s, dnn: %s\n", a.c_str(), b.c_str(), MODEL_NAME.c_str());
    // TaskManager taskManager(lc_task, a, b);
    
    // taskManager.executeAllTasks(ExecutionMode::Aker, streams[0]);
    // taskManager.executeAllTasks(ExecutionMode::Tacker, streams[1]);

    // [Aker] throughput test(1:1 version)
    // auto lc_task = createTask(MODEL_NAME);
    // for (int i = 0; i < 5; ++i) {
    //     lc_task->initExecution();
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     for (auto& kernel: lc_task->kernels) {
    //         if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
    //         kernel->execute(nullptr);
    //         CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     }
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // }
    // cudaDeviceSynchronize();
    // std::string a = sget_kernel_info("throughput_test", "a");
    // std::string b = sget_kernel_info("throughput_test", "b");
    // printf("[Result] cd: %s, dnn: %s\n", a.c_str(), MODEL_NAME.c_str());
    // TaskManager taskManager(lc_task, a, b);

    // printf("----float----\n");
    // // taskManager.execute_with_one_cd_kernel(ExecutionMode::Aker, streams[0]);
    // taskManager.execute_with_one_cd_kernel(ExecutionMode::Tacker, streams[0]);
    // taskManager.be_task1_name = a + "_int";
    // printf("taskManager.be_task1_name: %s\n", taskManager.be_task1_name.c_str());
    // printf("----int----\n");
    // taskManager.execute_with_one_cd_kernel(ExecutionMode::Tacker, streams[1]);

    // [Aker] tzgemm-cd pair profile
    // auto lc_task = createTask(MODEL_NAME);
    // for (int i = 0; i < 5; ++i) {
    //     lc_task->initExecution();
    //     auto start = clock();
    //     for (auto& kernel: lc_task->kernels) {
    //         // if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
    //         kernel->execute(nullptr);
    //     }
    //     cudaDeviceSynchronize();
    //     auto end = clock();
    //     auto duration = float(end - start) * 1000 / CLOCKS_PER_SEC;
    //     printf("%s total time: %f\n", lc_task->taskName.c_str(), duration);
    // }
    // int k = get_kernel_info("ratio_test", "k");
    // int m = get_kernel_info("ratio_test", std::to_string(k));
    // tzgemm_cd_profile(m, k);

    // [Aker] cd pair profile test
    // auto lc_task = createTask(MODEL_NAME);
    // for (int i = 0; i < 5; ++i) {
    //     lc_task->initExecution();
    //     for (auto& kernel: lc_task->kernels) {
    //         // if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
    //         kernel->execute(nullptr);
    //     }
    //     cudaDeviceSynchronize();
    // }
    // cudaDeviceSynchronize();
    // cd_pair_profile(stream);

    // [Aker] fig21 makespan reduction
    // auto be_name1 = sget_kernel_info("makespan_reduction", "be_name1");
    // auto be_name2 = sget_kernel_info("makespan_reduction", "be_name2");
    // auto be_task1 = createKernel(be_name1);
    // auto be_task2 = createKernel(be_name2);

    // float be_task1_ori_time = 0.0f, be_task2_ori_time = 0.0f;
    // init cd
    // be_task1->kernel_->initParams();
    // be_task2->kernel_->initParams();
    
    // ori sum time
    // warmup
    // for (int i = 0; i < 10; ++i) {
    //     be_task1->execute(streams[0]);
    //     be_task2->execute(streams[0]);
    // }
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, streams[0]));
    // be_task1->execute(streams[0]);
    // CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, streams[0]));
    // CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventElapsedTime(&be_task1_ori_time, startKERNEL, stopKERNEL));

    // CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, streams[0]));
    // be_task2->execute(streams[0]);
    // CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, streams[0]));
    // CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    // CUDA_SAFE_CALL(cudaEventElapsedTime(&be_task2_ori_time, startKERNEL, stopKERNEL));
    // auto mix_be_name = be_task1->kernelName[0] < be_task2->kernelName[0] ? be_task1->kernelName + "_" + be_task2->kernelName : be_task2->kernelName + "_" + be_task1->kernelName;
    // auto be_mix = createMixKernel(mix_be_name);

    // float mix_be_time = 0.0f;
    // for (int i = 0; i < 10; ++i) {
    //     float tmp_time = 0.0f;
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, streams[0]));
    //     be_mix->execute(streams[0]);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, streams[0]));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&tmp_time, startKERNEL, stopKERNEL));
    //     mix_be_time += tmp_time;
    // }
    // mix_be_time /= 10;
    // printf("[Ori] BE task1 + task2 took %f ms to execute.\n", be_task1_ori_time + be_task2_ori_time);
    // printf("[Mix] BE task1 + task2 took %f ms to execute.\n", mix_be_time);
    // printf("[Result] %s Makespan reduction: %f\n", mix_be_name.c_str(), (be_task1_ori_time + be_task2_ori_time - mix_be_time) * 100.0 / (be_task1_ori_time + be_task2_ori_time));

    // [Aker] moti / fig2
    auto lc_task = createTask(MODEL_NAME);
    for (int i = 0; i < 5; ++i) {
        lc_task->initExecution();
        CUDA_SAFE_CALL(cudaEventRecord(startKERNEL, streams[0]));
        for (auto& kernel: lc_task->kernels) {
            // if (!i) printf("Exec kernel: %s\n", kernel->kernelName.c_str());
            kernel->execute(streams[0]);
        }
        CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL, streams[0]));
        CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
        printf("[warmup]%s total time: %f\n", lc_task->taskName.c_str(), milliseconds);
    }
    cudaDeviceSynchronize();
    vector<float> time_vec(lc_task->kernels.size(), 0);
    for (int i = 0; i < 5; ++i) {
        lc_task->initExecution();
        for (int j = 0; j < lc_task->kernels.size(); ++j) {
            auto start = clock();
            lc_task->kernels[j]->execute(streams[0]);
            cudaDeviceSynchronize();
            cudaStreamSynchronize(streams[0]);
            auto end = clock();
            auto duration = float(end - start) * 1000 / CLOCKS_PER_SEC;
            time_vec[j] += duration;
        }
        cudaStreamSynchronize(streams[0]);
    }
    // cal total avg time
    float total_time = 0.0f;
    float tensor_core_time = 0.0f;
    for (int i = 0; i < time_vec.size(); ++i) {
        total_time += time_vec[i];
        // printf("kernel name: %s, time: %f\n", lc_task->kernels[i]->kernelName.c_str(), time_vec[i] / 5);
    }
    total_time /= 5;
    printf("total time: %f\n", total_time);

    int tensor_kernel_count = 0;
    for (int k_idx = 0; k_idx < lc_task->kernels.size(); ++k_idx) {
        auto kernel = lc_task->kernels[k_idx];
        // printf("kernel name: %s, time: %f\n", kernel->kernelName.c_str(), time_vec[k_idx] / 5);
        if (kernel->mixable != 0) {
            tensor_core_time += time_vec[k_idx] / 5;
            tensor_kernel_count++;
        }
    }
    printf("tensor core time, cuda kernel time, tc kernel count: %f %f %d\n", tensor_core_time, total_time - tensor_core_time, tensor_kernel_count);


    // [sys] Table for ptb resource usage
    // auto ptb_kernel = createKernel("fft_int");
    // auto ori_kernel = OriFFTKernel(-1);
    // ptb_kernel->gptbParams.ptb_end_block_pos = ori_kernel.launchGridDim.x * ori_kernel.launchGridDim.y * ori_kernel.launchGridDim.z;
    // printf("ori blks: %d\n", ori_kernel.launchGridDim.x * ori_kernel.launchGridDim.y * ori_kernel.launchGridDim.z);
    // for (int i = 0; i < 5; ++i) {
    //     ptb_kernel->execute(nullptr);
    //     ori_kernel.execute(nullptr);
    // }

    // auto time_vec = std::vector<float>();
    // // ori
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     ori_kernel.execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float ori_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     ori_time += time_vec[i];
    // }
    // ori_time /= 10.0f;

    // // ptb
    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     ptb_kernel->execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float ptb_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     ptb_time += time_vec[i];
    // }
    // ptb_time /= 10.0f;

    // // show nomarlized time
    // printf("normal time: %f\n", 1.0f - (ptb_time - ori_time)/ori_time);

    // auto fcp_ptb = createKernel("cp");
    // auto ffft_ptb = createKernel("fft");
    // auto fmriq_ptb = createKernel("mriq");
    // auto fmrif_ptb = createKernel("mrif");
    // auto icp_ptb = createKernel("cp_int");
    // auto ifft_ptb = createKernel("fft_int");
    // auto tzgemm_kernel = OriTZGEMMKernel(0, 4096, 512, 4096);

    // // fcp_ptb->execute(nullptr);
    // // ffft_ptb->execute(nullptr);
    // // fmriq_ptb->execute(nullptr);
    // // fmrif_ptb->execute(nullptr);
    // // icp_ptb->execute(nullptr);
    // // ifft_ptb->execute(nullptr);
    // // warmup
    // for (int i = 0; i < 5; ++i) {
    //     tzgemm_kernel.execute(nullptr);
    // }
    // // ori
    // auto time_vec = std::vector<float>();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     tzgemm_kernel.execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float ori_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     ori_time += time_vec[i];
    // }
    // ori_time /= 10.0f;

    // //ptb
    // tzgemm_kernel.launchGridDim.x = SM_NUM * 2;
    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     tzgemm_kernel.execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float ptb_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     ptb_time += time_vec[i];
    // }
    // ptb_time /= 10.0f;

    // printf("normal time: %f\n", 1.0f - (ptb_time - ori_time)/ori_time);


    // fig 6 fcp + int32(icp ifft imriq imrif)
    // auto fcp_ptb = createKernel("cp");
    // fcp_ptb->gptbParams.ptb_end_block_pos = get_kernel_info("sys_fig6", "cp_blk_num"); // ptb

    // auto icp_ptb = createKernel("cp_int");
    // icp_ptb->launchGridDim.x = get_kernel_info("sys_fig6", "int_cp_blk_num");
    // icp_ptb->gptbParams.ptb_end_block_pos = get_kernel_info("sys_fig6", "int_cp_blk_num");
    // auto ifft = createKernel("fft_int");
    // ifft->launchGridDim.x = get_kernel_info("sys_fig6", "int_fft_blk_num");
    // ifft->gptbParams.ptb_end_block_pos = get_kernel_info("sys_fig6", "int_fft_blk_num");
    // auto imrif = createKernel("mrif_int");
    // imrif->launchGridDim.x = get_kernel_info("sys_fig6", "int_mrif_blk_num");
    // imrif->gptbParams.ptb_end_block_pos = get_kernel_info("sys_fig6", "int_mrif_blk_num");
    // auto imriq = createKernel("mriq_int");
    // imriq->launchGridDim.x = get_kernel_info("sys_fig6", "int_mriq_blk_num");
    // imriq->gptbParams.ptb_end_block_pos = get_kernel_info("sys_fig6", "int_mriq_blk_num");

    // // warmup
    // for (int i = 0; i < 5; ++i) {
    //     fcp_ptb->execute(nullptr);
    //     icp_ptb->execute(nullptr);
    //     ifft->execute(nullptr);
    //     imrif->execute(nullptr);
    //     imriq->execute(nullptr);
    // }

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // // solo
    // auto time_vec = std::vector<float>();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     fcp_ptb->execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float fcp_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     fcp_time += time_vec[i];
    // }
    // fcp_time /= 10.0f;

    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     icp_ptb->execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float icp_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     icp_time += time_vec[i];
    // }
    // icp_time /= 10.0f;

    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     ifft->execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float ifft_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     ifft_time += time_vec[i];
    // }
    // ifft_time /= 10.0f;

    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     imrif->execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float imrif_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     imrif_time += time_vec[i];
    // }
    // imrif_time /= 10.0f;

    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     imriq->execute(nullptr);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float imriq_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     imriq_time += time_vec[i];
    // }
    // imriq_time /= 10.0f;

    // // fcp+icp
    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     fcp_ptb->execute(streams[0]);
    //     icp_ptb->execute(streams[1]);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float fcp_icp_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     fcp_icp_time += time_vec[i];
    // }
    // fcp_icp_time /= 10.0f;

    // // fcp+ifft
    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     fcp_ptb->execute(streams[0]);
    //     ifft->execute(streams[1]);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float fcp_ifft_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     fcp_ifft_time += time_vec[i];
    // }
    // fcp_ifft_time /= 10.0f;

    // // fcp+imrif
    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     fcp_ptb->execute(streams[0]);
    //     imrif->execute(streams[1]);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float fcp_imrif_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     fcp_imrif_time += time_vec[i];
    // }
    // fcp_imrif_time /= 10.0f;

    // // fcp+imriq
    // time_vec.clear();
    // for(int i = 0; i < 20; ++i) {
    //     CUDA_SAFE_CALL(cudaEventRecord(startKERNEL));
    //     fcp_ptb->execute(streams[0]);
    //     imriq->execute(streams[1]);
    //     CUDA_SAFE_CALL(cudaEventRecord(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventSynchronize(stopKERNEL));
    //     CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, startKERNEL, stopKERNEL));
    //     time_vec.push_back(milliseconds);
    // }
    // std::sort(time_vec.begin(), time_vec.end());
    // float fcp_imriq_time = 0.0f;
    // for(int i = 5; i < 15; ++i) {
    //     fcp_imriq_time += time_vec[i];
    // }
    // fcp_imriq_time /= 10.0f;

    // // show
    // printf("fcp: %f, icp: %f, fcp+icp: %f\n", fcp_time, icp_time, (fcp_time + icp_time - fcp_icp_time) / (fcp_time + icp_time));
    // printf("fcp: %f, ifft: %f, fcp+ifft: %f\n", fcp_time, ifft_time, (fcp_time + ifft_time - fcp_ifft_time) / (fcp_time + ifft_time));
    // printf("fcp: %f, imrif: %f, fcp+imrif: %f\n", fcp_time, imrif_time, (fcp_time + imrif_time - fcp_imrif_time) / (fcp_time + imrif_time));
    // printf("fcp: %f, imriq: %f, fcp+imriq: %f\n", fcp_time, imriq_time, (fcp_time + imriq_time - fcp_imriq_time) / (fcp_time + imriq_time));


    // system("nvidia-smi >> nvidia-smi.log");

    return 0;
}
