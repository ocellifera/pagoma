#
# CMake script
#

# Using modern cmake
cmake_minimum_required(VERSION 3.18)
# Project declaration
project(pagoma LANGUAGES CXX)

# Some brief user configuration options
option(PAGOMA_BUILD_EXAMPLES "Build example applications" ON)
option(PAGOMA_BUILD_TESTS "Build tests" ON)
option(PAGOMA_BUILD_STATIC_LIB "Build static library only" OFF)
option(PAGOMA_ENABLE_TIME_TRACE "Add -ftime-trace to Clang builds" OFF)
option(PAGOMA_USE_CUDA "Use CUDA backend to target NVIDIA GPUs" OFF)
option(PAGOMA_USE_HIP "Use HIP backend to target AMG GPUs" OFF)

# Global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For your IDE
set(CMAKE_CXX_STANDARD 20) # Our preferred c++ standard
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FetchContent for package consumption
include(FetchContent)

# CheckCXXSourceCompiles for checking that each package compiles
include(CheckCXXSourceCompiles)

# CheckCXXSourceRuns for checking that each package compiles & runs
include(CheckCXXSourceRuns)

# Find MPI with checks
include("cmake/modules/FindMPI.cmake")

# TODO: Add Kokkos fetch content

# Caliper for profiling
set(WITH_MPI on CACHE BOOL "" FORCE)
# TODO: Add cuda / rocm compilation
FetchContent_Declare(
  caliper
  GIT_REPOSITORY https://github.com/llnl/caliper.git
  GIT_TAG v2.13.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(caliper)

# p4est for octree mesh refinement
# TODO: Add fetch content for this

# deal.II for FEM
find_package(
  deal.II
  9.7.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not find a valid deal.II installation.")
endif()

# Test infrastructure
if(BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

if(NOT PAGOMA_BUILD_STATIC_LIB)
  enable_testing()
endif()
