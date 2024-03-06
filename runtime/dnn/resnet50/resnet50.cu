#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/resnet50/resnet50.h"
#include "./dnn/resnet50/resnet50_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void Resnet50::initParams() {
    cuda_init();
    //input argument
    float* Parameter_270_0_host, *Parameter_270_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_270_0_host, sizeof(float)* 150528));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_270_0, sizeof(float) * 150528));

    //output arguments
    float* Result_505_0_host, *Result_505_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_505_0_host, sizeof(float) * 1001));

    // fill input values
    for (int i = 0; i < 150528; ++i) Parameter_270_0_host[i] = 1.0f;

    CUDA_SAFE_CALL(cudaMemcpy(Parameter_270_0, Parameter_270_0_host, sizeof(float) * 150528, cudaMemcpyHostToDevice));
    this->Parameter_270_0 = Parameter_270_0;
    this->Parameter_270_0_host = Parameter_270_0_host;
    this->Result_505_0 = &Result_505_0;

    // put all kernel
    this->gen_vector(Parameter_270_0, &Result_505_0);
}