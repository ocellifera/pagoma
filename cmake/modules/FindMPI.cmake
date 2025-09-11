# Find the MPI package and make sure it compiles and runs some test
# code

# Find the package
find_package(MPI REQUIRED)

# Setup some variables for the compilation check
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
set(CMAKE_REQUIRED_INCLUDES ${MPI_CXX_INCLUDE_PATH})

set(
  TEST_CODE
  "
#include <mpi.h>
int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Finalize();
	return 0;
}
"
)

check_cxx_source_compiles("${TEST_CODE}" MPI_COMPILES)
check_cxx_source_runs("${TEST_CODE}" MPI_RUNS)

if(NOT MPI_COMPILES)
  message(
    FATAL_ERROR
    "MPI test compilation failed - check your MPI installation or compiler settings."
  )
endif()
if(NOT MPI_RUNS)
  message(
    FATAL_ERROR
    "MPI test run failed - check your MPI installation or compiler settings."
  )
endif()
