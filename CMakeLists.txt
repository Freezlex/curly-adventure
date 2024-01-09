cmake_minimum_required(VERSION 3.27)
project(curly_adventure CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(curly_adventure
        includes/kernel.h
        src/kernel.cpp
        src/kernel.cu
        src/main.cpp)

set_target_properties(curly_adventure PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
