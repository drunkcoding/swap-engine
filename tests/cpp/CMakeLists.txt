cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)

add_executable(test_model test_model.cc)
target_link_libraries(test_model ${TORCH_LIBRARIES})

# set debug flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
set_property(TARGET test_model PROPERTY CXX_STANDARD 17)