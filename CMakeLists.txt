cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_BUILD_TYPE Debug)

set(FLAGS)

project(SwapEngine LANGUAGES CXX CUDA)

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(gflags REQUIRED)

include_directories($ENV{HOME}/onnxruntime/include) 
include_directories(/usr/local/cuda-11.4/include) 
# link_directories($ENV{HOME}/onnxruntime/lib)

find_library(ORT_LIBRARY onnxruntime HINTS $ENV{HOME}/onnxruntime/lib)
find_library(CUDNN_LIBRARY libcudnn.so.8 HINTS /usr/local/cuda-11.4/lib64 /usr/lib/x86_64-linux-gnu)
find_library(CUDART_LIBRARY cudart HINTS /usr/local/cuda-11.4/lib64 /usr/lib/x86_64-linux-gnu)

add_executable(SwapEngine main.cc)
target_link_libraries(SwapEngine gflags ${ORT_LIBRARY} ${CUDNN_LIBRARY} ${CUDART_LIBRARY})