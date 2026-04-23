cmake_minimum_required(VERSION 3.26)
project(sad_torch LANGUAGES CXX)

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS deployment version")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

if(DEFINED Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Development Development.SABIModule Interpreter)
  if (NOT Python3_FOUND)
    message(FATAL_ERROR "Unable to find python matching: ${Python3_EXECUTABLE}.")
  endif()
else()
  find_package(Python3 REQUIRED COMPONENTS Development Development.SABIModule Interpreter)
endif()

append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
find_package(Torch REQUIRED)

# Optional Metal backend (macOS only).
option(SAD_USE_METAL "Build Metal backend" ON)
set(USE_METAL OFF)
if(APPLE AND SAD_USE_METAL)
  enable_language(OBJCXX)
  set(USE_METAL ON)
endif()

# Optional CUDA backend (Linux/Windows).
option(SAD_USE_CUDA "Build CUDA backend" ON)
set(USE_CUDA OFF)
if(SAD_USE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(USE_CUDA ON)
  endif()
endif()

if(NOT USE_METAL AND NOT USE_CUDA)
  message(FATAL_ERROR "Neither Metal nor CUDA backend is enabled/available.")
endif()

set(SRC
  ${CMAKE_CURRENT_LIST_DIR}/torch_binding.cpp
)

if(USE_CUDA)
  list(APPEND SRC
    ${CMAKE_CURRENT_LIST_DIR}/cuda/sad_bindings.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/render.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/candidate_pack.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/jfa.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/gradients.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/tau_diffusion.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/adam.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/densify_prune.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/utils.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src/radix_sort.cu
  )
endif()

if(USE_METAL)
  list(APPEND SRC ${CMAKE_CURRENT_LIST_DIR}/metal/sad_bindings.mm)
endif()

Python3_add_library(_sad_ext MODULE WITH_SOABI ${SRC})
set_target_properties(_sad_ext PROPERTIES
  CXX_STANDARD 17
  PREFIX ""
)

# Expose the extension name to C++.
target_compile_definitions(_sad_ext PRIVATE
  "TORCH_EXTENSION_NAME=_sad_ext"
)

# Include dirs (torch headers + local).
target_include_directories(_sad_ext PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/../../cuda/src
)

# Link torch.
target_link_libraries(_sad_ext PRIVATE torch ${TORCH_LIBRARIES})

if(USE_METAL)
  target_compile_definitions(_sad_ext PRIVATE "METAL_KERNEL")
  target_compile_options(_sad_ext PRIVATE
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
  )
  target_link_libraries(_sad_ext PRIVATE "-framework Metal" "-framework Foundation")

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/compile-metal.cmake)
  set(METAL_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/metal/torch_sad.metal"
    "${CMAKE_CURRENT_LIST_DIR}/../../metal/shaders/sad.metal"
    "${CMAKE_CURRENT_LIST_DIR}/../../metal/shaders/radix_sort.metal"
  )
  compile_metal_shaders(_sad_ext
    "${METAL_SOURCES}"
    "${CMAKE_CURRENT_LIST_DIR}/../../metal/shaders"
  )
endif()

if(USE_CUDA)
  target_compile_definitions(_sad_ext PRIVATE "CUDA_KERNEL")
  set_target_properties(_sad_ext PROPERTIES CUDA_STANDARD 17)
  target_link_libraries(_sad_ext PRIVATE CUDA::cudart)
endif()
