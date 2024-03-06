#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/inception3/inception3.h"
#include "./dnn/inception3/inception3_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void Inception3::initParams() {
    cuda_init();
    //input argument
    float* Parameter_485_0_host, *Parameter_485_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_485_0_host, sizeof(float)* 268203));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_485_0, sizeof(float) * 268203));

    //output arguments
    float* Result_892_0_host, *Result_892_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_892_0_host, sizeof(float) * 1001));

    // fill input values
    for (int i = 0; i < 268203; ++i) Parameter_485_0_host[i] = 1.0f;

    CUDA_SAFE_CALL(cudaMemcpy(Parameter_485_0, Parameter_485_0_host, sizeof(float) * 268203, cudaMemcpyHostToDevice));

    this->Parameter_485_0 = Parameter_485_0;
    this->Parameter_485_0_host = Parameter_485_0_host;
    this->Result_892_0 = &Result_892_0;
    // put all kernel
    this->gen_vector(Parameter_485_0, &Result_892_0);
}

