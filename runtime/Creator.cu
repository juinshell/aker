#include "Creator.h"
#include "Logger.h"

extern Logger logger;


int toUnicode(const char* str)
{
	return str[0] + (str[1] != '\0' ? toUnicode(str + 1) : 0);
}

constexpr inline int myHash(const char* str)
{
	return str[0] + (str[1] != '\0' ? myHash(str + 1) : 0);
}

unordered_map<std::string, GPTBKernel*> kernelMap;


GPTBKernel* createKernel(const std::string &name) {
    switch (toUnicode(name.c_str())) {
        case myHash("cp"):
            if (kernelMap.find("cp") == kernelMap.end()) {
                // printf("[Creator] create cp kernel\n");
                kernelMap["cp"] = new GPTBKernel(
                    10, 
                    "cp",
                    "gptb_cp", 
                    new OriCPKernel(10), 
                    dim3(SM_NUM * 6, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    32 * 512);
            } 
            return kernelMap["cp"];
            break;
        case myHash("cutcp"):
            if (kernelMap.find("cutcp") == kernelMap.end()) {
                kernelMap["cutcp"] = new GPTBKernel(
                    11, 
                    "cutcp",
                    "gptb_cutcp", 
                    new OriCUTCPKernel(11), 
                    dim3(SM_NUM * 6, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    1352);
            }
            return kernelMap["cutcp"];
            break;
        case myHash("fft"):
            if (kernelMap.find("fft") == kernelMap.end()) {
                kernelMap["fft"] = new GPTBKernel(
                    12, 
                    "fft",
                    "gptb_fft", 
                    new OriFFTKernel(12), 
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    10240);
            }
            return kernelMap["fft"];
        case myHash("lbm"):
            if (kernelMap.find("lbm") == kernelMap.end()) {
                kernelMap["lbm"] = new GPTBKernel(
                    16, 
                    "lbm",
                    "gptb_lbm", 
                    new OriLBMKernel(16), 
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    16384);
            }
            return kernelMap["lbm"];
        case myHash("mrif"):
            if (kernelMap.find("mrif") == kernelMap.end()) {
                kernelMap["mrif"] = new GPTBKernel(
                    17, 
                    "mrif",
                    "gptb_mrif", 
                    new OriMRIFKernel(17), 
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    1024);
            }
            return kernelMap["mrif"];
        case myHash("mriq"):
            if (kernelMap.find("mriq") == kernelMap.end()) {
                kernelMap["mriq"] = new GPTBKernel(
                    18, 
                    "mriq",
                    "gptb_mriq", 
                    new OriMRIQKernel(18), 
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(256, 1, 1), 
                    0, 
                    819);
            }
            return kernelMap["mriq"];
        case myHash("sgemm"):
            if (kernelMap.find("sgemm") == kernelMap.end()) {
                kernelMap["sgemm"] = new GPTBKernel(
                    19, 
                    "sgemm",
                    "gptb_sgemm", 
                    new OriSGEMMKernel(19), 
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    774);
            }
            return kernelMap["sgemm"];
        case myHash("stencil"):
            if (kernelMap.find("stencil") == kernelMap.end()) {
                kernelMap["stencil"] = new GPTBKernel(
                    20, 
                    "stencil",
                    "gptb_stencil", 
                    new OriSTENCILKernel(20), 
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(128, 1, 1), 
                    0, 
                    1024);
            }
            return kernelMap["stencil"];
        default:
            logger.ERROR("Creator: Kernel not found: " + name);
        // case myHash("cutcp"): 
        //     return new GPTBKernel(
        //         11, 
        //         "cutcp",
        //         "gptb_cutcp", 
        //         new OriCUTCPKernel(11), 
        //         dim3(SM_NUM * 6, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         1352);
        // case myHash("fft"):
        //     return new GPTBKernel(
        //         12, 
        //         "fft",
        //         "gptb_fft", 
        //         new OriFFTKernel(12), 
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         10240);
        // case myHash("lbm"):
        //     return new GPTBKernel(
        //         16, 
        //         "lbm",
        //         "gptb_lbm", 
        //         new OriLBMKernel(16), 
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         16384);
        // case myHash("mrif"):
        //     return new GPTBKernel(
        //         17, 
        //         "mrif",
        //         "gptb_mrif", 
        //         new OriMRIFKernel(17), 
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(256, 1, 1), 
        //         0, 
        //         1024);
        // case myHash("mriq"):
        //     return new GPTBKernel(
        //         18, 
        //         "mriq",
        //         "gptb_mriq", 
        //         new OriMRIQKernel(18), 
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(256, 1, 1), 
        //         0, 
        //         819);
        // case myHash("sgemm"):
        //     return new GPTBKernel(
        //         19, 
        //         "sgemm",
        //         "gptb_sgemm", 
        //         new OriSGEMMKernel(19), 
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         774);
        // case myHash("stencil"):
        //     return new GPTBKernel(
        //         20, 
        //         "stencil",
        //         "gptb_stencil", 
        //         new OriSTENCILKernel(20), 
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(128, 1, 1), 
        //         0, 
        //         1024);
    }
}

unordered_map<std::string, MixKernel* > mixKernelMap;

MixKernel* createMixKernel(const std::string &name) {
    switch (myHash(name.c_str())) {
        case myHash("cp_fft"):
            // printf("[Creator] hit cp_fft kernel\n");
            if (mixKernelMap.find("cp_fft") == mixKernelMap.end()) {
                mixKernelMap["cp_fft"] = new MixKernel(
                    0, 
                    "cp_fft", 
                    createKernel("cp"),
                    createKernel("fft"),
                    dim3(SM_NUM * 2, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("cp")->gptbParams.ptb_start_block_pos,
                    createKernel("cp")->gptbParams.ptb_end_block_pos, 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cp_fft"];
        case myHash("cp_sgemm"):
            if (mixKernelMap.find("cp_sgemm") == mixKernelMap.end()) {
                mixKernelMap["cp_sgemm"] = new MixKernel(
                    1, 
                    "cp_sgemm", 
                    createKernel("cp"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("cp")->gptbParams.ptb_start_block_pos,
                    createKernel("cp")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cp_sgemm"];
        case myHash("cutcp_fft"):
            if (mixKernelMap.find("cutcp_fft") == mixKernelMap.end()) {
                mixKernelMap["cutcp_fft"] = new MixKernel(
                    2, 
                    "cutcp_fft", 
                    createKernel("cutcp"),
                    createKernel("fft"),
                    dim3(SM_NUM * 3, 1, 1), 
                    dim3(786, 1, 1), 
                    createKernel("cutcp")->gptbParams.ptb_start_block_pos,
                    createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cutcp_fft"];
        case myHash("cutcp_sgemm"):
            if (mixKernelMap.find("cutcp_sgemm") == mixKernelMap.end()) {
                mixKernelMap["cutcp_sgemm"] = new MixKernel(
                    3, 
                    "cutcp_sgemm", 
                    createKernel("cutcp"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 4, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("cutcp")->gptbParams.ptb_start_block_pos,
                    createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["cutcp_sgemm"];
        case myHash("fft_lbm"):
            if (mixKernelMap.find("fft_lbm") == mixKernelMap.end()) {
                mixKernelMap["fft_lbm"] = new MixKernel(
                    4, 
                    "fft_lbm", 
                    createKernel("fft"),
                    createKernel("lbm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(906, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_lbm"];
        case myHash("fft_mriq"):
            if (mixKernelMap.find("fft_mriq") == mixKernelMap.end()) {
                mixKernelMap["fft_mriq"] = new MixKernel(
                    5, 
                    "fft_mriq", 
                    createKernel("fft"),
                    createKernel("mriq"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(896, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("mriq")->gptbParams.ptb_start_block_pos,
                    createKernel("mriq")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_mriq"];
        case myHash("fft_sgemm"):
            if (mixKernelMap.find("fft_sgemm") == mixKernelMap.end()) {
                mixKernelMap["fft_sgemm"] = new MixKernel(
                    6, 
                    "fft_sgemm", 
                    createKernel("fft"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(640, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_sgemm"];
        case myHash("lbm_mrif"):
            if (mixKernelMap.find("lbm_mrif") == mixKernelMap.end()) {
                mixKernelMap["lbm_mrif"] = new MixKernel(
                    7, 
                    "lbm_mrif", 
                    createKernel("lbm"),
                    createKernel("mrif"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(896, 1, 1), 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos, 
                    createKernel("mrif")->gptbParams.ptb_start_block_pos,
                    createKernel("mrif")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lbm_mrif"];
        case myHash("lbm_mriq"):
            if (mixKernelMap.find("lbm_mriq") == mixKernelMap.end()) {
                mixKernelMap["lbm_mriq"] = new MixKernel(
                    8, 
                    "lbm_mriq", 
                    createKernel("lbm"),
                    createKernel("mriq"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(640, 1, 1), 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos, 
                    createKernel("mriq")->gptbParams.ptb_start_block_pos,
                    createKernel("mriq")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lbm_mriq"];
        case myHash("lbm_sgemm"):
            if (mixKernelMap.find("lbm_sgemm") == mixKernelMap.end()) {
                mixKernelMap["lbm_sgemm"] = new MixKernel(
                    9, 
                    "lbm_sgemm", 
                    createKernel("lbm"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("lbm")->gptbParams.ptb_start_block_pos,
                    createKernel("lbm")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["lbm_sgemm"];
        case myHash("mrif_sgemm"):
            if (mixKernelMap.find("mrif_sgemm") == mixKernelMap.end()) {
                mixKernelMap["mrif_sgemm"] = new MixKernel(
                    10, 
                    "mrif_sgemm", 
                    createKernel("mrif"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(768, 1, 1), 
                    createKernel("mrif")->gptbParams.ptb_start_block_pos,
                    createKernel("mrif")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["mrif_sgemm"];
        case myHash("mriq_sgemm"):
            if (mixKernelMap.find("mriq_sgemm") == mixKernelMap.end()) {
                mixKernelMap["mriq_sgemm"] = new MixKernel(
                    11, 
                    "mriq_sgemm", 
                    createKernel("mriq"),
                    createKernel("sgemm"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(768, 1, 1), 
                    createKernel("mriq")->gptbParams.ptb_start_block_pos,
                    createKernel("mriq")->gptbParams.ptb_end_block_pos, 
                    createKernel("sgemm")->gptbParams.ptb_start_block_pos,
                    createKernel("sgemm")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["mriq_sgemm"];
        case myHash("fft_stencil"):
            if (mixKernelMap.find("fft_stencil") == mixKernelMap.end()) {
                mixKernelMap["fft_stencil"] = new MixKernel(
                    12, 
                    "fft_stencil", 
                    createKernel("fft"),
                    createKernel("stencil"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("fft")->gptbParams.ptb_start_block_pos,
                    createKernel("fft")->gptbParams.ptb_end_block_pos, 
                    createKernel("stencil")->gptbParams.ptb_start_block_pos,
                    createKernel("stencil")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["fft_stencil"];
        case myHash("mrif_stencil"):
            if (mixKernelMap.find("mrif_stencil") == mixKernelMap.end()) {
                mixKernelMap["mrif_stencil"] = new MixKernel(
                    13, 
                    "mrif_stencil", 
                    createKernel("mrif"),
                    createKernel("stencil"),
                    dim3(SM_NUM * 1, 1, 1), 
                    dim3(1024, 1, 1), 
                    createKernel("mrif")->gptbParams.ptb_start_block_pos,
                    createKernel("mrif")->gptbParams.ptb_end_block_pos, 
                    createKernel("stencil")->gptbParams.ptb_start_block_pos,
                    createKernel("stencil")->gptbParams.ptb_end_block_pos);
            }
            return mixKernelMap["mrif_stencil"];
        default:
            logger.ERROR("Creator: Kernel not found: " + name);
        // case myHash("cp_fft"):
        //     return new MixKernel(
        //         0, 
        //         "cp_fft", 
        //         (cp),
        //         (fft),
        //         dim3(SM_NUM * 2, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("cp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos);
        // case myHash("cp_sgemm"):
        //     return new MixKernel(
        //         1, 
        //         "cp_sgemm", 
        //         (cp),
        //         (sgemm),
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("cp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // case myHash("cutcp_fft"):
        //     return new MixKernel(
        //         2, 
        //         "cutcp_fft", 
        //         (cutcp),
        //         (fft),
        //         dim3(SM_NUM * 3, 1, 1), 
        //         dim3(786, 1, 1), 
        //         createKernel("cutcp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos);
        // case myHash("cutcp_sgemm"):
        //     return new MixKernel(
        //         3, 
        //         "cutcp_sgemm", 
        //         (cutcp),
        //         (sgemm),
        //         dim3(SM_NUM * 4, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("cutcp")->gptbParams.ptb_start_block_pos,
        //         createKernel("cutcp")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // case myHash("fft_lbm"):
        //     return new MixKernel(
        //         4, 
        //         "fft_lbm", 
        //         (fft),
        //         (lbm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(906, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos);
        // case myHash("fft_mriq"):
        //     return new MixKernel(
        //         5, 
        //         "fft_mriq", 
        //         (fft),
        //         (mriq),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(896, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("mriq")->gptbParams.ptb_start_block_pos,
        //         createKernel("mriq")->gptbParams.ptb_end_block_pos);
        // case myHash("fft_sgemm"):
        //     return new MixKernel(
        //         6, 
        //         "fft_sgemm", 
        //         (fft),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(640, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // case myHash("lbm_mrif"):
        //     return new MixKernel(
        //         7, 
        //         "lbm_mrif", 
        //         (lbm),
        //         (mrif),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(896, 1, 1), 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos, 
        //         createKernel("mrif")->gptbParams.ptb_start_block_pos,
        //         createKernel("mrif")->gptbParams.ptb_end_block_pos);
        // case myHash("lbm_mriq"):
        //     return new MixKernel(
        //         8, 
        //         "lbm_mriq", 
        //         (lbm),
        //         (mriq),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(640, 1, 1), 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos, 
        //         createKernel("mriq")->gptbParams.ptb_start_block_pos,
        //         createKernel("mriq")->gptbParams.ptb_end_block_pos);
        // case myHash("lbm_sgemm"):
        //     return new MixKernel(
        //         9, 
        //         "lbm_sgemm", 
        //         (lbm),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("lbm")->gptbParams.ptb_start_block_pos,
        //         createKernel("lbm")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // case myHash("mrif_sgemm"):
        //     return new MixKernel(
        //         10, 
        //         "mrif_sgemm", 
        //         (mrif),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(768, 1, 1), 
        //         createKernel("mrif")->gptbParams.ptb_start_block_pos,
        //         createKernel("mrif")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // case myHash("mriq_sgemm"):
        //     return new MixKernel(
        //         11, 
        //         "mriq_sgemm", 
        //         (mriq),
        //         (sgemm),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(768, 1, 1), 
        //         createKernel("mriq")->gptbParams.ptb_start_block_pos,
        //         createKernel("mriq")->gptbParams.ptb_end_block_pos, 
        //         createKernel("sgemm")->gptbParams.ptb_start_block_pos,
        //         createKernel("sgemm")->gptbParams.ptb_end_block_pos);
        // case myHash("fft_stencil"):
        //     return new MixKernel(
        //         12, 
        //         "fft_stencil", 
        //         (fft),
        //         (stencil),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("fft")->gptbParams.ptb_start_block_pos,
        //         createKernel("fft")->gptbParams.ptb_end_block_pos, 
        //         createKernel("stencil")->gptbParams.ptb_start_block_pos,
        //         createKernel("stencil")->gptbParams.ptb_end_block_pos);
        // case myHash("mrif_stencil"):
        //     return new MixKernel(
        //         13, 
        //         "mrif_stencil", 
        //         (mrif),
        //         (stencil),
        //         dim3(SM_NUM * 1, 1, 1), 
        //         dim3(1024, 1, 1), 
        //         createKernel("mrif")->gptbParams.ptb_start_block_pos,
        //         createKernel("mrif")->gptbParams.ptb_end_block_pos, 
        //         createKernel("stencil")->gptbParams.ptb_start_block_pos,
        //         createKernel("stencil")->gptbParams.ptb_end_block_pos);

    }
}