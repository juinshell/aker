# CMakeLists.txt
cmake_minimum_required (VERSION 3.8)
project (tacker VERSION 0.3.1 LANGUAGES C CXX CUDA)
add_compile_options(-std=c++11)

find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_LIBRARIES: " ${CUDA_LIBRARIES})

SET(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")

set(CUDA_NVCC_FLAGS "-arch=sm_75 -Xptxas -dlcm=ca --ptxas-options=-v ${CUDA_NVCC_FLAGS}") # must have -arch=sm_75

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if (NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING 
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
set(CMAKE_CXX_FLAGS "-O3  -lcuda -lcublas -lcurand ${CMAKE_CXX_FLAGS}") # must have -lcuda, linker is /usr/bin/c++, not nvcc
message(STATUS "${PROJECT_NAME} compile cuda code in [RELEASE] mode")
else()
set(CMAKE_CXX_FLAGS "-G -O0 -lcuda -lcublas -lcurand ${CMAKE_CXX_FLAGS}")
message(STATUS "${PROJECT_NAME} compile cuda code in [DEBUG] mode")
endif()

set(LOG_FILE_PATH ${PROJECT_SOURCE_DIR}/log/${PROJECT_VERSION}.log)

# # compile library
# add_library(readatom OBJECT ${PROJECT_SOURCE_DIR}/file_t/readatom.c)    
# target_include_directories(readatom PUBLIC ${PROJECT_SOURCE_DIR}/file_t/header)

set(SRC_LIST main.cu Logger.cc TaskManager.cu Task.cu ModuleCenter.cu readatom.c Recorder.cu)

string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)
configure_file(TackerConfig.h.in TackerConfig.h)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST})

set_target_properties( ${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


target_include_directories(${PROJECT_NAME} PUBLIC
                           ${PROJECT_BINARY_DIR}
                           ${CUDA_INCLUDE_DIRS}
                           ${PROJECT_SOURCE_DIR}
                          #  ../mix_kernels
                          #  ../mix_kernels/file_t/header
                           )  
add_definitions(-w)