#!/bin/bash
#
# This script formats CMake files so we have a semblance of standard
# formatting. We use gersemi (which is pip installable), which can be
# found at https://github.com/BlankSpruce/gersemi
#
# Usage:
#   ./contrib/utilities/cmake_format.sh
#

if test ! -d src -o ! -d include; then
	echo "This script must be run from the top-level directory of pagoma"
	exit 1
fi

if ! [ -x "$(command -v gersemi)" ]; then
	echo "Make sure gersemi is in the path"
	exit 2
fi

# To format the files we have to be a bit careful because gersemi
# will recursively look for cmake files. This means if we simply
# specify the directory `.` it will also format the dependencies!
gersemi -i "CMakeLists.txt"
gersemi -i "cmake"
gersemi -i "src"
gersemi -i "test" 

echo "Done"
exit 0
