// _Z28mixed_cutcp_sgemm_kernel_1_1iiP6float4fffPfiiiiiiiiiiS1_S1_S1_iiiiiiiiiiii
#include <mma.h>
using namespace nvcuda; 
#include "header/atom.h"
#include "cutcp_sgemm_kernel.h"
#include "Logger.h"
#include "header/cutcp_header.h"
#include "header/sgemm_header.h"
#include "util.h"
#include "TackerConfig.h"
#include "ModuleCenter.h"

extern Logger logger;
extern ModuleCenter moduleCenter;

// 构造函数
CUTCP_SGEMM_Kernel::CUTCP_SGEMM_Kernel(int id, const std::string& name, CUTCP_SGEMM_ParamsStruct& params) {
    Id = id;
    kernelName = name;
    initParams();
}

CUTCP_SGEMM_Kernel::CUTCP_SGEMM_Kernel(int id, const std::string& name){
    Id = id;
    kernelName = name;
    initParams();
}

// 初始化参数default
void CUTCP_SGEMM_Kernel::initParams() {
      int cutcp_blks = 6;
    int cutcp_iter = 1;
    Atoms *atom;
    LatticeDim lattice_dim;
    Lattice *gpu_lattice;
    Vec3 min_ext, max_ext;	    /* Bounding box of atoms */
    Vec3 lo, hi;			    /* Bounding box with padding  */
    float h = 0.5f;		        /* Lattice spacing */
    float cutoff = 12.f;		/* Cutoff radius */
    float padding = 0.5f;		/* Bounding box padding distance */

    const char *pqrfilename = "/home/jxdeng/workspace/tacker/0_mybench/file_t/cutcp_input.pqr";
    if (!(atom = read_atom_file(pqrfilename))) {
        // fprintf(stderr, "read_atom_file() failed\\n");
        logger.ERROR("read_atom_file() failed");
        exit(1);
    }
    get_atom_extent(&min_ext, &max_ext, atom);
    lo = (Vec3) {min_ext.x - padding, min_ext.y - padding, min_ext.z - padding};
    hi = (Vec3) {max_ext.x + padding, max_ext.y + padding, max_ext.z + padding};
    lattice_dim = lattice_from_bounding_box(lo, hi, h);
    gpu_lattice = create_lattice(lattice_dim);

    float4 *binBaseAddr;
    int3 *nbrlist;
    nbrlist = (int3 *)malloc(NBRLIST_MAXLEN * sizeof(int3));
    int nbins = 32768;
    binBaseAddr = (float4 *) calloc(nbins * BIN_DEPTH, sizeof(float4));
    prepare_input(gpu_lattice, cutoff, atom, binBaseAddr, nbrlist);

    int nbrlistlen = 256;
    // float *cutcp_ori_regionZeroCuda, *host_cutcp_ori_regionZeroCuda;
    // float4 *cutcp_ori_binBaseCuda, *cutcp_ori_binZeroCuda;
    // float *cutcp_ptb_regionZeroCuda, *host_cutcp_ptb_regionZeroCuda;
    // float4 *cutcp_ptb_binBaseCuda, *cutcp_ptb_binZeroCuda;
    float *cutcp_gptb_regionZeroCuda, *host_cutcp_gptb_regionZeroCuda;
    float4 *cutcp_gptb_binBaseCuda, *cutcp_gptb_binZeroCuda;

    int lnx = 208;
    int lny = 208;
    int lnz = 208;
    int lnall = lnx * lny * lnz;

    int xRegionDim = 26;
    int yRegionDim = 26;
    int zRegionDim = 26;
    int binDim_x = 32;
    int binDim_y = 32;
    float cutoff2 = 144.0;
    float inv_cutoff2 = 0.006944;

    // CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_ori_regionZeroCuda, lnall * sizeof(float)));
    // CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_ptb_regionZeroCuda, lnall * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_gptb_regionZeroCuda, lnall * sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemset(cutcp_ori_regionZeroCuda, 0, lnall * sizeof(float)));
    // CUDA_SAFE_CALL(cudaMemset(cutcp_ptb_regionZeroCuda, 0, lnall * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemset(cutcp_gptb_regionZeroCuda, 0, lnall * sizeof(float)));

    // CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_ori_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
    // CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_ptb_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
    CUDA_SAFE_CALL(cudaMalloc((void **) &cutcp_gptb_binBaseCuda, nbins * BIN_DEPTH * sizeof(float4)));
    // CUDA_SAFE_CALL(cudaMemcpy(cutcp_ori_binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
    //     cudaMemcpyHostToDevice));
    // CUDA_SAFE_CALL(cudaMemcpy(cutcp_ptb_binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
    //     cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cutcp_gptb_binBaseCuda, binBaseAddr, nbins * BIN_DEPTH * sizeof(float4),
        cudaMemcpyHostToDevice));

    // cutcp_ori_binZeroCuda = cutcp_ori_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;
    // cutcp_ptb_binZeroCuda = cutcp_ptb_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;
    cutcp_gptb_binZeroCuda = cutcp_gptb_binBaseCuda + ((3 * binDim_y + 3) * binDim_x + 3) * BIN_DEPTH;

    // host_cutcp_ori_regionZeroCuda = (float *)malloc(lnall * sizeof(float));
    // host_cutcp_ptb_regionZeroCuda = (float *)malloc(lnall * sizeof(float));
    host_cutcp_gptb_regionZeroCuda = (float *)malloc(lnall * sizeof(float));

    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrListLen, &nbrlistlen, sizeof(int), 0));
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(NbrList, nbrlist, nbrlistlen * sizeof(int3), 0));

    dim3 cutcp_grid, cutcp_block, ori_cutcp_grid, ori_cutcp_block;
    cutcp_grid.x = xRegionDim;
    cutcp_grid.y = yRegionDim;
    cutcp_grid.z = cutcp_iter * 2;
    cutcp_block.x = 8;
    cutcp_block.y = 2;
    cutcp_block.z = 8;
    ori_cutcp_grid = cutcp_grid;
    ori_cutcp_block = cutcp_block;


    this->kernelParams = new CUTCP_SGEMM_ParamsStruct();
    this->kernelParams->cutcp0_binDim_x = binDim_x;
    this->kernelParams->cutcp0_binDim_y = binDim_y;
    this->kernelParams->cutcp0_binZeroAddr = cutcp_gptb_binZeroCuda;
    this->kernelParams->cutcp0_h = h;
    this->kernelParams->cutcp0_cutoff2 = cutoff2;
    this->kernelParams->cutcp0_inv_cutoff2 = inv_cutoff2;
    this->kernelParams->cutcp0_regionZeroAddr = cutcp_gptb_regionZeroCuda;
    this->kernelParams->cutcp0_zRegionIndex_t = 25;
    this->kernelParams->cutcp0_grid_dimension_x = ori_cutcp_grid.x;
    this->kernelParams->cutcp0_grid_dimension_y = ori_cutcp_grid.y;
    this->kernelParams->cutcp0_grid_dimension_z = ori_cutcp_grid.z;
    this->kernelParams->cutcp0_block_dimension_x = ori_cutcp_block.x;
    this->kernelParams->cutcp0_block_dimension_y = ori_cutcp_block.y;
    this->kernelParams->cutcp0_block_dimension_z = ori_cutcp_block.z;
    this->kernelParams->cutcp0_ptb_start_block_pos = 0; // TODO
    this->kernelParams->cutcp0_ptb_iter_block_step = launchGridDim.x * launchGridDim.y * launchGridDim.z; // TODO
    this->kernelParams->cutcp0_ptb_end_block_pos = 0; // TODO


    this->loadKernel();
}

// 虚析构函数实现
CUTCP_SGEMM_Kernel::~CUTCP_SGEMM_Kernel() {
  
    
    delete this->kernelParams;

    // logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is destroyed!");
}

void CUTCP_SGEMM_Kernel::execute() {
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

void CUTCP_SGEMM_Kernel::loadKernel() {
    // Implementation of CP kernel load logic here
    logger.INFO("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " is loading ...");

    this->function = moduleCenter.getFunction("cutcp_sgemm", "_Z28mixed_cutcp_sgemm_kernel_1_1iiP6float4fffPfiiiiiiiiiiS1_S1_S1_iiiiiiiiiiii");

    if (this->function == nullptr) {
        logger.ERROR("kernel name: " + kernelName + ", id: " + std::to_string(Id) + " load failed!");
        exit(EXIT_FAILURE);
    }

    return ;
}
