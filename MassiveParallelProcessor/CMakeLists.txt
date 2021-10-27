cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_Practice CXX C CUDA)
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

add_subdirectory(Add)
add_subdirectory(imgProcess)
add_subdirectory(Multi/Multi-baseline)
add_subdirectory(Multi/Multi-tiled)
add_subdirectory(Conv)
