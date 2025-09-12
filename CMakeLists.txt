#
# CMake script
#

# Using modern cmake
cmake_minimum_required(VERSION 3.18)
# Project declaration
project(pagoma LANGUAGES CXX)

# Check for in-source builds
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "In-source builds are not allowed.")
endif()

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

# ExternalProject for package consumption
include(ExternalProject)

# CheckCXXSourceCompiles for checking that each package compiles
include(CheckCXXSourceCompiles)

# CheckCXXSourceRuns for checking that each package compiles & runs
include(CheckCXXSourceRuns)

# Find MPI with checks
include("cmake/modules/FindMPI.cmake")

# Find Kokkos with checks
include("cmake/modules/FindKokkos.cmake")

# Find zlib with checks
include("cmake/modules/FindZLIB.cmake")

# Find p4est with checks
include("cmake/modules/FindP4EST.cmake")

# Find deal.II with checks

# Find Caliper with checks

# Grab the source files
set(PAGOMA_SOURCE
pagoma.cc
)

# Create the library
if(PAGOMA_BUILD_STATIC_LIB)
	add_library(pagoma STATIC ${PAGOMA_SOURCE})
else()
	add_library(pagoma SHARED ${PAGOMA_SOURCE})
endif()

# Link the dependencies to our library
target_link_libraries(pagoma PUBLIC MPI::MPI_CXX)
