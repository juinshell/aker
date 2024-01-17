#pragma once
#include "GPTBKernel.h"

extern Logger logger;
extern std::unordered_map<std::string, void*> fmap;

class MixKernel : public Kernel {
public:
    MixKernel(int id, const std::string& funcKey,
        std::unique_ptr<GPTBKernel> kernel1, std::unique_ptr<GPTBKernel> kernel2, dim3 gridDim, dim3 blockDim, 
        int ptb_start_block_pos1, int ptb_end_block_pos1, int ptb_start_block_pos2, int ptb_end_block_pos2)
        : kernel1(std::move(kernel1)), kernel2(std::move(kernel2)), funcKey(funcKey){
            this->Id = id;
            this->kernelName = funcKey;
            this->launchGridDim = gridDim;
            this->launchBlockDim = blockDim;

            this->smem = this->kernel1->smem + this->kernel2->smem;

            // override gptb params
            this->kernel1->gptbParams.ptb_iter_block_step = gridDim.x * gridDim.y * gridDim.z;
            this->kernel2->gptbParams.ptb_iter_block_step = this->kernel1->gptbParams.ptb_iter_block_step;

            this->kernel1->gptbParams.ptb_start_block_pos = ptb_start_block_pos1;
            this->kernel1->gptbParams.ptb_end_block_pos = ptb_end_block_pos1;
            this->kernel2->gptbParams.ptb_start_block_pos = ptb_start_block_pos2;
            this->kernel2->gptbParams.ptb_end_block_pos = ptb_end_block_pos2;

            initParams();
            loadKernel();
    }

    ~MixKernel() {
        // logger.INFO("id: " + std::to_string(Id) + " is destroyed!");
    }

    const std::string funcKey;

    // TODO extra kernel if divided need

    void initParams() override{
        // kernel1 args push
        for (auto &param : kernel1->kernelParams) {
            kernelParams.push_back(param);
        }
        // kernel overrided args push
        kernelParams.push_back(&kernel1->gptbParams.ptb_start_block_pos);
        kernelParams.push_back(&kernel1->gptbParams.ptb_iter_block_step);
        kernelParams.push_back(&kernel1->gptbParams.ptb_end_block_pos);

        // kernel2 args push
        for (auto &param : kernel2->kernelParams) {
            kernelParams.push_back(param);
        }
        // kernel overrided args push
        kernelParams.push_back(&kernel2->gptbParams.ptb_start_block_pos);
        kernelParams.push_back(&kernel2->gptbParams.ptb_iter_block_step);
        kernelParams.push_back(&kernel2->gptbParams.ptb_end_block_pos);
    }

    void loadKernel() override {
        if (fmap.find(funcKey) != fmap.end()) {
            this->kernelFunc = (void*)fmap[funcKey];
        } else {
            logger.ERROR("load kernel {" + funcKey + "} failed!");
            exit(EXIT_FAILURE);
        }

        return ;
    }

    void execute() override {
        CUDA_SAFE_CALL(cudaLaunchKernel(this->kernelFunc, 
            launchGridDim, launchBlockDim,
            (void **)kernelParams.data(), (size_t)this->smem, 0));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

private:
    std::unique_ptr<GPTBKernel> kernel1;
    std::unique_ptr<GPTBKernel> kernel2;

    GPTBKernel* extraKernel;
};
