#
# Macro to add collect hints for a package. First we grab the cache
# variable, then the environment variable.
#
# Usage:
#   package_hints(package _var)
#

macro(package_hints package _var)
  string(TOUPPER "${package}" _package_upper)
  set(_hints "")

  # Check the cache variable
  if(${_package_upper}_DIR)
    list(APPEND _hints "${${_package_upper}_DIR}")
  elseif(DEFINED ENV{${_package_upper}_DIR})
    list(APPEND _hints "$ENV{${_package_upper}_DIR}")
  endif()

  set(${_var} "${_hints}")
endmacro()
