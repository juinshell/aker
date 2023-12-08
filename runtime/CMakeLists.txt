# CMakeLists.txt
cmake_minimum_required (VERSION 3.2)
project (tacker VERSION 0.1.0 LANGUAGES CXX CUDA)
add_compile_options(-std=c++11)

SET(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")


set(CMAKE_CXX_FLAGS "-g -O0 ${CMAKE_CXX_FLAGS}")

# cuda
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)
find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
message(STATUS "cuda include: " ${CUDA_INCLUDE_DIRS})
message(STATUS "cuda lib: " ${CUDA_LIBRARIES})
message(STATUS "CMAKE_SHARED_LIBRARY_LINK_CUDA_FLAGS: " ${CMAKE_SHARED_LIBRARY_LINK_CUDA_FLAGS})

# # CMAKE_CXX_FLAGS添加-lcuda -lcudart
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcuda -lcudart -lcudadevrt -lcudart_static")

include_directories(${CUDA_INCLUDE_DIRS})

set(LOG_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/log/${PROJECT_VERSION}.log)

set(SRC_LIST main.cu cp_kernel.cu Logger.cc Task.cc TaskManager.cc)
string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)

configure_file(TackerConfig.h.in TackerConfig.h)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST})

target_include_directories(${PROJECT_NAME} PUBLIC
                           ${PROJECT_BINARY_DIR}
                           ${CUDA_LIBRARIES}
                           )