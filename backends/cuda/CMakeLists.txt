cmake_minimum_required(VERSION 3.20)
project(sad_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(sad_cuda
    src/sad_cuda.cu
    src/jfa.cu
    src/gradients.cu
    src/render.cu
    src/adam.cu
    src/utils.cu
    src/candidate_pack.cu
    src/densify_prune.cu
    src/radix_sort.cu
    src/tau_diffusion.cu
)

target_include_directories(sad_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

set_target_properties(sad_cuda PROPERTIES
    CUDA_ARCHITECTURES "86"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
)

if (MSVC)
    target_compile_options(sad_cuda PRIVATE /W4 $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
else()
    target_compile_options(sad_cuda PRIVATE -Wall -Wextra -Wno-unused-parameter $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()
