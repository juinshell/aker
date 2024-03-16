#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/vgg16/vgg16.h"
#include "./dnn/vgg16/vgg16_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void VGG16::initParams() {
    vgg16_cuda_init();
    //input argument
    float* Parameter_0_0_host, *Parameter_0_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_0_0_host, sizeof(float)* 2408448));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_0_0, sizeof(float) * 2408448));

    //output arguments
    float* Result_144_0_host, *Result_144_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_144_0_host, sizeof(float) * 16016));

    // fill input values
    for (int i = 0; i < 2408448; ++i) Parameter_0_0_host[i] = 1.0f;

    CUDA_SAFE_CALL(cudaMemcpy(Parameter_0_0, Parameter_0_0_host, sizeof(float) * 2408448, cudaMemcpyHostToDevice));
    this->Parameter_0_0 = Parameter_0_0;
    this->Parameter_0_0_host = Parameter_0_0_host;
    this->Result_144_0 = &Result_144_0;

    // put all kernel
    this->gen_vector(Parameter_0_0, &Result_144_0);
}