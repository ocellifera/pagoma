# Find the Kokkos package and make sure it compiles and runs some test
# code

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_HIP OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA_CONSTEXPR OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE OFF CACHE BOOL "" FORCE)

if(PAGOMA_USE_CUDA)
  set(Kokkos_ENABLE_CUDA ON)
  set(Kokkos_ENABLE_CUDA_CONSTEXPR ON)
  if(PAGOMA_BUILD_STATIC_LIB)
    set(Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE ON)
  endif()
elseif(PAGOMA_USE_HIP)
  set(Kokkos_ENABLE_HIP ON)
  set(Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTATIONS ON)
  if(PAGOMA_BUILD_STATIC_LIB)
    set(Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE ON)
  endif()
endif()

if(PAGOMA_EXTERNAL_PACKAGES_ONLY)
  find_package(Kokkos 4.4.00 REQUIRED CONFIG)
elseif(PAGOMA_EMBEDDED_PACKAGES_ONLY)
  FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG 4.7.00
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(Kokkos)
else()
  find_package(Kokkos 4.4.00 CONFIG)
  if(Kokkos_FOUND)
    message(
      STATUS
      "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")"
    )
  else()
    FetchContent_Declare(
      Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.7.00
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()
