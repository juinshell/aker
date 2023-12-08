// cp_kernel.cu
#include "cp_kernel.h"
#include "Logger.h"
#include "header/cp_header.h"
#include "util.h"
#include "TackerConfig.h"
#include <cuda.h>
#include "cuda.h"
#include <cuda_runtime.h>

extern Logger logger;

// 构造函数
OriCPKernel::OriCPKernel(int id, const std::string& name, OriCPParamsStruct& params) {
    kernelId = id;
    kernelName = name;
    initParams();
}

OriCPKernel::OriCPKernel(int id, const std::string& name){
    kernelId = id;
    kernelName = name;
    initParams();
}

// 初始化参数default
void OriCPKernel::initParams() {
    int cp_blks = 8;
    int cp_iter = 1;
    float *atoms = NULL;
    int atomcount = ATOMCOUNT;
    const float gridspacing = 0.1;					// number of atoms to simulate
    dim3 volsize(VOLSIZEX, VOLSIZEY, 1);
    initatoms(&atoms, atomcount, volsize, gridspacing);

    // allocate and initialize the GPU output array
    int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

    float *ori_output;	
    // float *ptb_output;
    // float *gptb_output;
    CUDA_SAFE_CALL(cudaMalloc((void**)&ori_output, volmemsz));
    CUDA_SAFE_CALL(cudaMemset(ori_output, 0, volmemsz));
    // cudaErrCheck(cudaMalloc((void**)&ptb_output, volmemsz));
    // cudaErrCheck(cudaMemset(ptb_output, 0, volmemsz));
    // cudaErrCheck(cudaMalloc((void**)&gptb_output, volmemsz));
    // cudaErrCheck(cudaMemset(gptb_output, 0, volmemsz));
    float *host_ori_energy = (float *) malloc(volmemsz);
    // float *host_ptb_energy = (float *) malloc(volmemsz);
    // float *host_gptb_energy = (float *) malloc(volmemsz);

    dim3 cp_grid, cp_block;
    int atomstart = 1;
    int runatoms = MAXATOMS;
    // ---------------------------------------------------------------------------------------

    // SOLO running
    // ---------------------------------------------------------------------------------------
    cp_block.x = BLOCKSIZEX;						// each thread does multiple Xs
    cp_block.y = BLOCKSIZEY;
    cp_block.z = 1;
    cp_grid.x = volsize.x / (cp_block.x * UNROLLX); // each thread does multiple Xs
    cp_grid.y = volsize.y / cp_block.y; 
    cp_grid.z = volsize.z / cp_block.z; 

    copyatomstoconstbuf(atoms + 4 * atomstart, runatoms, 0*gridspacing);

    this->kernelParams = new OriCPParamsStruct();
    this->kernelParams->numatoms = runatoms;
    this->kernelParams->gridspacing = 0.1;
    this->kernelParams->energygrid = ori_output;
    this->kernelParams->iteration = 1;
    this->launchGridDim = cp_grid;
    this->launchBlockDim = cp_block;
    this->loadKernel();
}

// 虚析构函数实现
OriCPKernel::~OriCPKernel() {
    CU_SAFE_CALL(cuModuleUnload(this->module));
    CUDA_SAFE_CALL(cudaFree(this->kernelParams->energygrid));
    delete this->kernelParams;

    logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(kernelId) + " is destroyed!");
}

void OriCPKernel::execute() {
    // Implementation of CP kernel execution logic here
    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(kernelId) + " is executing ...");
    // // print CPParamsStruct parameters by this->kernelParams
    // logger.INFO("numatoms: " + std::to_string(this->kernelParams->numatoms));
    // logger.INFO("gridspacing: " + std::to_string(this->kernelParams->gridspacing));
    // logger.INFO("energygrid: " + std::to_string((uint64_t)this->kernelParams->energygrid));
    // logger.INFO("iteration: " + std::to_string(this->kernelParams->iteration));

    void *launchargs[] = {(void *)&this->kernelParams->numatoms, (void *)&this->kernelParams->gridspacing, (void *)&this->kernelParams->energygrid, (void *)&this->kernelParams->iteration};
    CU_SAFE_CALL(cuLaunchKernel(this->function, 
    launchGridDim.x, launchGridDim.y, launchGridDim.z, 
    launchBlockDim.x, launchBlockDim.y, launchBlockDim.z, 
    0, NULL, launchargs, NULL));

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
}

void OriCPKernel::loadKernel() {
    // Implementation of CP kernel load logic here
    logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(kernelId) + " is loading ...");

    std::string module_file = std::string(CMAKELISTS_PATH) + std::string("/cubins/ori_cp.cubin");
    // logger.INFO("module_file: " + std::string(module_file));

	dim3 block, grid;

    CU_SAFE_CALL(cuModuleLoad(&this->module, module_file.c_str()));

	const char* cdkernel_name = "_Z6ori_cpifPfi";
	CU_SAFE_CALL(cuModuleGetFunction(&this->function, this->module, cdkernel_name));

    return ;
}
