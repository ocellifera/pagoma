#
# CMake script
#

set(TARGET "pagoma")

set(TARGET_SRC
${CMAKE_SOURCE_DIR}/include/utilities.h
${CMAKE_SOURCE_DIR}/include/invm.h
${CMAKE_SOURCE_DIR}/include/mesh_manager.h
${TARGET}.cc
)

cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.7.0 HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
if(NOT ${deal.II_FOUND})
	message(FATAL_ERROR "Could not find a valid deal.II installation.")
endif()

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()
