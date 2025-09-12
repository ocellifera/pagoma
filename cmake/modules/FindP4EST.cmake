# Find the p4est package and make sure it compiles and runs some test
# code

if(PAGOMA_EXTERNAL_PACKAGES_ONLY)
	find_package(P4EST 2.8.7 REQUIRED)
elseif(PAGOMA_EMBEDDED_PACKAGES_ONLY)
	FetchContent_Declare(
				p4est
				GIT_REPOSITORY https://github.com/cburstedde/p4est.git
				GIT_TAG v2.8.7
				GIT_SHALLOW TRUE
		)
FetchContent_MakeAvailable(p4est)
else()
	find_package(P4EST 2.8.7)
	if(P4EST_FOUND)
		message(
						STATUS
						"Found p4est: ${P4EST_DIR} (version \"${P4EST_VERSION}\")"
				)
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


