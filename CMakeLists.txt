cmake_minimum_required(VERSION 3.13)

project(um LANGUAGES CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)

add_executable(triad triad.cu)
 set_target_properties(triad PROPERTIES
   CUDA_STANDARD 11
   CUDA_STANDARD_REQUIRED ON
   CUDA_EXTENSIONS OFF
 )
