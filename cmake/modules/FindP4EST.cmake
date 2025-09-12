# Find the p4est package and make sure it compiles and runs some test
# code

set(P4EST_DIR "" CACHE PATH "An optional hint to a p4est installation")
package_hints(p4est _p4est_hints)

if(PAGOMA_EXTERNAL_PACKAGES_ONLY)
  find_package(P4EST 2.3.6 REQUIRED HINTS ${_p4est_hints})
elseif(PAGOMA_EMBEDDED_PACKAGES_ONLY)
  FetchContent_Declare(
    p4est
    GIT_REPOSITORY https://github.com/cburstedde/p4est.git
    GIT_TAG v2.8.7
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(p4est)
else()
  find_package(P4EST 2.3.6 HINTS ${_p4est_hints})
  if(P4EST_FOUND)
    message(STATUS "Found p4est: ${P4EST_DIR} (version \"${P4EST_VERSION}\")")
  else()
    FetchContent_Declare(
      p4est
      GIT_REPOSITORY https://github.com/cburstedde/p4est.git
      GIT_TAG v2.8.7
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(p4est)
  endif()
endif()
