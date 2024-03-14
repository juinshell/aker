#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/vgg11/vgg11.h"
#include "./dnn/vgg11/vgg11_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void VGG11::initParams() {
    cuda_init();
    //input argument
    float* Parameter_32_0_host, *Parameter_32_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_32_0_host, sizeof(float)* 4816896));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_32_0, sizeof(float) * 4816896));

    //output arguments
    float* Result_99_0_host, *Result_99_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_99_0_host, sizeof(float) * 32032));

    // fill input values
    for (int i = 0; i < 4816896; ++i) Parameter_32_0_host[i] = 1.0f;

    CUDA_SAFE_CALL(cudaMemcpy(Parameter_32_0, Parameter_32_0_host, sizeof(float) * 4816896, cudaMemcpyHostToDevice));
    this->Parameter_32_0 = Parameter_32_0;
    this->Parameter_32_0_host = Parameter_32_0_host;
    this->Result_99_0 = &Result_99_0;

    // put all kernel
    this->gen_vector(Parameter_32_0, &Result_99_0);
}