# CMakeLists.txt
cmake_minimum_required (VERSION 3.8)
project (tacker VERSION 0.4.0 LANGUAGES C CXX CUDA)
add_compile_options(-std=c++11)

find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_LIBRARIES: " ${CUDA_LIBRARIES})

SET(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")

set(CUDA_NVCC_FLAGS "-arch=sm_75 -Xptxas -dlcm=ca --ptxas-options=-v -cudart shared ${CUDA_NVCC_FLAGS}") # must have -arch=sm_75, -cudart, not -lcudart

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if (NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING 
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

if(${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
set(CMAKE_CXX_FLAGS " -lcuda -lcublas -lcurand ${CMAKE_CXX_FLAGS}") # must have -lcuda, linker is /usr/bin/c++, not nvcc. must no -lcudart
message(STATUS "${PROJECT_NAME} compile cuda code in [RELEASE] mode")
else()
set(CMAKE_CXX_FLAGS "-g -lcuda -lcublas -lcurand ${CMAKE_CXX_FLAGS}")
message(STATUS "${PROJECT_NAME} compile cuda code in [DEBUG] mode")
endif()

set(LOG_FILE_PATH ${PROJECT_SOURCE_DIR}/log/${PROJECT_VERSION}.log)

# cudnn
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)
include_directories(${CUDNN_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

set(SRC_LIST main.cu Logger.cc TaskManager.cu Task.cu ModuleCenter.cu readatom.c Recorder.cc ./dnn/resnet50/resnet50.cu json.cc Creator.cu)

string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)
configure_file(TackerConfig.h.in TackerConfig.h)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST})

# set_target_properties( ${PROJECT_NAME}
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
#                        )

target_link_options(${PROJECT_NAME} PRIVATE -Xptxas -wl)

# # hook lib
# add_library(mylib  ./hook/mylib.cu)
# target_compile_options(mylib PRIVATE -I/usr/local/cuda/include -Xcompiler -fPIC -shared -ldl -L/usr/local/cuda/lib64 -lcudart -arch=sm_75 -lcublas -lcurand)
target_include_directories(${PROJECT_NAME} PUBLIC
                           ${PROJECT_BINARY_DIR}
                           ${CUDA_INCLUDE_DIRS}
                           ${PROJECT_SOURCE_DIR}
                           ${CUDNN_LIBRARIES}
                        #    mylib
                          #  ../mix_kernels
                          #  ../mix_kernels/file_t/header
                           )  
target_link_libraries(${PROJECT_NAME} cudnn) 