# CMakeLists.txt
cmake_minimum_required (VERSION 3.10)
project (tacker VERSION 0.2.0 LANGUAGES CUDA CXX)
add_compile_options(-std=c++11)

SET(CMAKE_CXX_COMPILER "/usr/local/cuda/bin/nvcc")


set(CMAKE_CXX_FLAGS "-g -O0 -lcuda ${CMAKE_CXX_FLAGS}")

# cuda
# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.6)

# if(NOT DEFINED CMAKE_CUDA_STANDARD)
#     set(CMAKE_CUDA_STANDARD 11)
#     set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# endif()

find_package(CUDA REQUIRED)
message(STATUS "cuda version: " ${CUDA_VERSION_STRING})
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_LIBRARIES: " ${CUDA_LIBRARIES})
message(STATUS "CUDA_CUDART_LIBRARY: " ${CUDA_CUDART_LIBRARY})
message(STATUS "CUDA_LIBRARY_DIRS: " ${CUDA_LIBRARY_DIRS})
message(STATUS "CMAKE_SHARED_LIBRARY_LINK_CUDA_FLAGS: " ${CMAKE_SHARED_LIBRARY_LINK_CUDA_FLAGS})

# # CMAKE_CXX_FLAGS添加-lcuda -lcudart
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcuda -lcudart -lcudadevrt -lcudart_static")

set(LOG_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/log/${PROJECT_VERSION}.log)

set(SRC_LIST main.cu cp_kernel.cu Logger.cc Task.cc TaskManager.cc)
string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)

configure_file(TackerConfig.h.in TackerConfig.h)

CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST})

# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "75")

target_include_directories(${PROJECT_NAME} PUBLIC
                           ${PROJECT_BINARY_DIR}
                           ${CUDA_INCLUDE_DIRS}
                           )                
                                                           # /usr/bin/c++ CMakeFiles/tacker.dir/Logger.cc.o CMakeFiles/tacker.dir/Task.cc.o CMakeFiles/tacker.dir/TaskManager.cc.o CMakeFiles/tacker.dir/tacker_generated_main.cu.o CMakeFiles/tacker.dir/tacker_generated_cp_kernel.cu.o -o tacker   -L/usr/local/cuda-12/targets/x86_64-linux/lib/stubs  -L/usr/local/cuda-12/targets/x86_64-linux/lib  /usr/local/cuda-12/lib64/libcudart_static.a -ldl /usr/lib/x86_64-linux-gnu/librt.a /usr/local/cuda-12/lib64/libcudart_static.a -ldl /usr/lib/x86_64-linux-gnu/librt.a -lcudadevrt -lcudart_static -lrt -lpthread -ldl
# target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES}) # /usr/bin/c++ CMakeFiles/tacker.dir/Logger.cc.o CMakeFiles/tacker.dir/Task.cc.o CMakeFiles/tacker.dir/TaskManager.cc.o CMakeFiles/tacker.dir/tacker_generated_main.cu.o CMakeFiles/tacker.dir/tacker_generated_cp_kernel.cu.o -o tacker   -L/usr/local/cuda-12/targets/x86_64-linux/lib/stubs  -L/usr/local/cuda-12/targets/x86_64-linux/lib  /usr/local/cuda-12/lib64/libcudart_static.a -ldl /usr/lib/x86_64-linux-gnu/librt.a /usr/local/cuda-12/lib64/libcudart_static.a -ldl /usr/lib/x86_64-linux-gnu/librt.a -lcudadevrt -lcudart_static -lrt -lpthread -ldl