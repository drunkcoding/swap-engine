cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)

add_executable(test_model test_model.cc)
target_link_libraries(test_model ${TORCH_LIBRARIES})

# set debug flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
set_property(TARGET test_model PROPERTY CXX_STANDARD 17)

add_executable(test_memory test_memory.cc)
target_link_libraries(test_memory pthread)

# include cuda path
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include)
include_directories(${CUDA_INCLUDE_DIRS})

# include cuda library
set(CUDA_LIBRARIES /usr/local/cuda/lib64)
link_directories(${CUDA_LIBRARIES})

set_source_files_properties(test_cuda_copy.cpp PROPERTIES LANGUAGE CUDA)
add_executable(test_cuda_copy test_cuda_copy.cc)
target_link_libraries(test_cuda_copy pthread cudart)