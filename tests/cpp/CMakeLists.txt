cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)

include(FetchContent)
FetchContent_Declare(
  rmm
  GIT_REPOSITORY https://github.com/rapidsai/rmm.git
  GIT_TAG v22.10.00
)
FetchContent_GetProperties(rmm)
if (NOT rmm_POPULATED)
  FetchContent_Populate(rmm)
  # add_subdirectory(${rmm_SOURCE_DIR} ${magic_enum_BINARY_DIR})
  message(STATUS "rmm_SOURCE_DIR: ${rmm_SOURCE_DIR}/include")
  include_directories(${rmm_SOURCE_DIR}/include)
endif ()
FetchContent_MakeAvailable(rmm)

# set debug flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")

link_directories(/home/xly/.conda/envs/swap-engine/lib)

add_executable(test_model test_model.cc)
target_link_libraries(test_model ${TORCH_LIBRARIES})
set_property(TARGET test_model PROPERTY CXX_STANDARD 17)

add_executable(test_unified_memory test_unified_memory.cc)
target_link_libraries(test_unified_memory ${TORCH_LIBRARIES})
set_property(TARGET test_unified_memory PROPERTY CXX_STANDARD 17)

add_executable(test_memory test_memory.cc)
target_link_libraries(test_memory pthread)


add_executable(test_copy test_copy.cc)
target_include_directories(test_copy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_copy pthread ${TORCH_LIBRARIES} cudart)
set_property(TARGET test_copy PROPERTY CXX_STANDARD 17)

# # include cuda path
# set(CUDA_INCLUDE_DIRS /usr/local/cuda-11.7/include)
# include_directories(${CUDA_INCLUDE_DIRS})

# # include cuda library
# set(CUDA_LIBRARIES /usr/local/cuda-11.7/lib64)
# link_directories(${CUDA_LIBRARIES})

# set_source_files_properties(test_cuda_copy.cpp PROPERTIES LANGUAGE CUDA)
# add_executable(test_cuda_copy test_cuda_copy.cc)
# target_link_libraries(test_cuda_copy pthread cudart)