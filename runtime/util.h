// util.h
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>

#define CU_SAFE_CALL(err) __checkCudaErrors(err, __FILE__, __LINE__)
// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) \
{                                                                             \
  if (CUDA_SUCCESS != err)                                           \
  {                                                                           \
    const char *errorStr = NULL;                                              \
    cuGetErrorString(err, &errorStr);                                         \
    fprintf(stderr, "CU_SAFE_CALL() Driver API error = %04d \"%s\" from file <%s>, line %i.\n", err, errorStr, file, line); \
    exit(EXIT_FAILURE);                                                       \
  }                                                                           \
}

#define checkKernelErrors(expr)                             \
  do {                                                      \
    expr;                                                   \
                                                            \
    cudaError_t __err = cudaGetLastError();                 \
    if (__err != cudaSuccess) {                             \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
             cudaGetErrorString(__err));                    \
      abort();                                              \
    }                                                       \
  } while (0)

#define CUDA_SAFE_CALL(x)                                                                   \
  do                                                                                        \
  {                                                                                         \
    cudaError_t result = (x);                                                               \
    if (result != cudaSuccess)                                                              \
    {                                                                                       \
      const char *msg = cudaGetErrorString(result);                                         \
      std::stringstream safe_call_ss;                                                       \
      safe_call_ss << "\nerror: " #x " failed with error"                                   \
                   << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
      throw std::runtime_error(safe_call_ss.str());                                         \
    }                                                                                       \
  } while (0)