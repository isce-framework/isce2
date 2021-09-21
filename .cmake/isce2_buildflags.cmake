# TODO (global build flags)
# These definitions and compile options are
# set globally for convenience.
# Perhaps we should apply them only as needed on a
# per-target basis, and propagate them via the interface?
add_definitions(-DNEEDS_F77_TRANSLATION -DF77EXTERNS_LOWERCASE_TRAILINGBAR)
add_compile_options(
    $<$<COMPILE_LANGUAGE:Fortran>:-ffixed-line-length-none>
    $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>
    $<$<COMPILE_LANGUAGE:Fortran>:-fno-range-check>
    $<$<COMPILE_LANGUAGE:Fortran>:-fno-second-underscore>)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:Fortran>:-fallow-argument-mismatch>)
endif()

# Set up build flags for C++ and Fortran.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED y)
set(CMAKE_CXX_EXTENSIONS n)

include(GNUInstallDirs)

# add automatically determined parts of the RPATH, which point to directories
# outside of the build tree, to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# the RPATH to be used when installing, but only if it's not a system directory
set(abs_libdir ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES ${abs_libdir} isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
    list(APPEND CMAKE_INSTALL_RPATH ${abs_libdir})
endif()

option(ISCE2_STRICT_COMPILATION "Enable strict checks during compilation" ON)
if(ISCE2_STRICT_COMPILATION)

    # Set -fno-common when supported to catch ODR violations
    include(CheckCCompilerFlag)
    check_c_compiler_flag(-fno-common C_FNO_COMMON)
    if(C_FNO_COMMON)
        add_compile_options($<$<COMPILE_LANGUAGE:C>:-fno-common>)
    endif()
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-fno-common CXX_FNO_COMMON)
    if(CXX_FNO_COMMON)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-common>)
    endif()
endif()
