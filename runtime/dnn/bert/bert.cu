#include "Logger.h"
#include "util.h"
#include "TackerConfig.h"
#include "Recorder.h"
#include "./dnn/bert/bert.h"
#include "./dnn/bert/bert_kernel_class.h"

extern Logger logger;
extern Recorder recorder;

void Bert::initParams() {
    cuda_init();
    //input argument
    int32_t* Parameter_964_0_host, *Parameter_964_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_964_0_host, sizeof(int32_t)* 512));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_964_0, sizeof(int32_t) * 512));
    //input argument
    int32_t* Parameter_965_0_host, *Parameter_965_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_965_0_host, sizeof(int32_t)* 512));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_965_0, sizeof(int32_t) * 512));
    //input argument
    int32_t* Parameter_966_0_host, *Parameter_966_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Parameter_966_0_host, sizeof(int32_t)* 512));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Parameter_966_0, sizeof(int32_t) * 512));

    //output arguments
    float* Result_3689_0_host, *Result_3689_0;
    CUDA_SAFE_CALL(cudaMallocHost((void**)&Result_3689_0_host, sizeof(float) * 1001));

    // fill input values
    for (int i = 0; i < 512; ++i) Parameter_964_0_host[i] = 1.0f;
    for (int i = 0; i < 512; ++i) Parameter_965_0_host[i] = 1.0f;
    for (int i = 0; i < 512; ++i) Parameter_966_0_host[i] = 1.0f;

    CUDA_SAFE_CALL(cudaMemcpy(Parameter_964_0, Parameter_964_0_host, sizeof(int32_t) * 512, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_965_0, Parameter_965_0_host, sizeof(int32_t) * 512, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Parameter_966_0, Parameter_966_0_host, sizeof(int32_t) * 512, cudaMemcpyHostToDevice));

    this->Parameter_964_0 = Parameter_964_0;
    this->Parameter_965_0 = Parameter_965_0;
    this->Parameter_966_0 = Parameter_966_0;

    this->Parameter_964_0_host = Parameter_964_0_host;
    this->Parameter_965_0_host = Parameter_965_0_host;
    this->Parameter_966_0_host = Parameter_966_0_host;

    this->Result_3689_0 = &Result_3689_0;

    // put all kernel
    this->gen_vector(Parameter_964_0, Parameter_965_0, Parameter_966_0, &Result_3689_0);
}

