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

# Global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For your IDE
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# FetchContent for package consumption
include(FetchContent)

# CheckCXXSourceCompiles for checking each package
include(CheckCXXSourceCompiles)

# Find MPI with checks
find_package(MPI REQUIRED)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH})

check_cxx_source_compiles("
	#include <mpi.h>
	int main(int argc, char** argv) {
		MPI_Init(&argc, &argv);
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Finalize();
		return 0;
	}"
MPI_WORKS)

if(NOT MPI_WORKS)
	message(FATAL_ERROR "MPI test compilation failed — check your MPI installation or compiler settings.")
endif()

# Caliper for profiling
set(with_mpi on cache bool "" force)
fetchcontent_declare(caliper
	git_repository https://github.com/llnl/caliper.git
	git_tag v2.13.1
	git_shallow true
)
fetchcontent_makeavailable(caliper)

set(cmake_required_libraries caliper)
check_cxx_source_compiles("
	#include <caliper/cali.h>
	int main() {
		CALI_CXX_MARK_FUNCTION;
		CALI_MARK_BEGIN('computing');
    		int ret = 2 * 3 * 7;
		CALI_MARK_END('computing');
    		return ret;
	}"
caliper_works)

if(NOT CALIPER_WORKS)
	message(FATAL_ERROR "Caliper test compilation failed — check your Caliper installation.")
endif()

# Test infrastructure
if(BUILD_TESTS)
	FetchContent_Declare(googletest
		GIT_REPOSITORY https://github.com/google/googletest.git
		GIT_TAG v1.17.0
		GIT_SHALLOW TRUE
	)
	FetchContent_MakeAvailable(googletest)
endif()
