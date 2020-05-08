# TODO (global build flags)
# These definitions and compile options are
# set globally for convenience.
# Perhaps we should apply them only as needed on a
# per-target basis, and propagate them via the interface?
add_definitions(-DNEEDS_F77_TRANSLATION -DF77EXTERNS_LOWERCASE_TRAILINGBAR)
add_compile_options(
    $<$<COMPILE_LANGUAGE:Fortran>:-ffixed-line-length-none>
    $<$<COMPILE_LANGUAGE:Fortran>:-fno-range-check>
    $<$<COMPILE_LANGUAGE:Fortran>:-fno-second-underscore>)

# Set up build flags for C++ and Fortran.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED y)
set(CMAKE_CXX_EXTENSIONS n)

# TODO (fix RPATHs)
# We have to hack our RPATHs a bit for these shared libraries to be
# loaded by others on the install-side. Maybe these libraries should
# be combined and/or installed to a common ISCE2 lib directory.
# Is there a semantic way to propagate their RPATHs
# without using these global variables?
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
list(APPEND CMAKE_INSTALL_RPATH
    ${CMAKE_INSTALL_PREFIX}/${ISCE2_PKG}/components/isceobj/Util
    )
