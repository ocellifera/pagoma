# Find the Kokkos package and make sure it compiles and runs some test
# code

if(PAGOMA_EXTERNAL_PACKAGES_ONLY)
	find_package(Kokkos 4.4.00 REQUIRED CONFIG)
elseif(PAGOMA_EMBEDDED_PACKAGES_ONLY)
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
