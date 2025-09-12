# Find the deal.II package and make sure it compiles and runs some test
# code

set(DEAL_II_DIR "" CACHE PATH "An optional hint to a deal.II installation")
package_hints(DEAL_II _deal_ii_hints)

if(PAGOMA_EXTERNAL_PACKAGES_ONLY)
  find_package(deal.II 9.7.0 REQUIRED HINTS ${_deal_ii_hints})
elseif(PAGOMA_EMBEDDED_PACKAGES_ONLY)
  FetchContent_Declare(
    deal.II
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG v9.7.0
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(deal.II)
else()
  find_package(deal.II 9.7.0 HINTS ${_deal_ii_hints})
  if(deal.II_FOUND)
    message(
      STATUS
      "Found deal.II: ${deal.II_DIR} (version \"${deal.II_VERSION}\")"
    )
  else()
    FetchContent_Declare(
      deal.II
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG v9.7.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(deal.II)
  endif()
endif()
