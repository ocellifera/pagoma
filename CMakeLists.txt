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
option(
  PAGOMA_EXTERNAL_PACKAGES_ONLY
  "Try to use external packages whenever possible"
  OFF
)
option(
  PAGOMA_EMBEDDED_PACKAGES_ONLY
  "Try to use embedded packages whenever possible"
  OFF
)
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

# Find Kokkos with checks
include("cmake/modules/FindKokkos.cmake")
