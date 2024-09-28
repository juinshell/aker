#include "util.h"
#include "cp_kernel.h"
#include "cutcp_kernel.h"
#include "fft_kernel.h"
#include "lbm_kernel.h"
#include "mrif_kernel.h"
#include "mriq_kernel.h"
#include "sgemm_kernel.h"
#include "stencil_kernel.h"
#include "tzgemm_kernel.h"

#include "lava_kernel.h"
#include "hot3d_kernel.h"
#include "nn_kernel.h"
#include "path_kernel.h"

#include "GPTBKernel.h"
#include "MixKernel.h"

#include <unordered_map>

using namespace std;

constexpr uint64_t FNV_prime = 16777619u;
constexpr uint64_t FNV_offset_basis = 2166136261u;

constexpr inline int myHash(const char* text);

GPTBKernel* createKernel(const std::string &name);

MixKernel* createMixKernel(const std::string &name);