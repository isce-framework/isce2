#[[
Usage:
  find_package(FFTW [REQUIRED] [QUIET] [COMPONENTS ...])

Be warned that this will only search for FFTW3 libraries.

It sets the following variables:
  FFTW_FOUND                 .. true if FFTW is found on the system
  FFTW_[component]_LIB_FOUND .. true if the component is found (see below)
  FFTW_LIBRARIES             .. full paths to all found FFTW libraries
  FFTW_[component]_LIB       .. full path to one component (see below)
  FFTW_INCLUDE_DIRS          .. FFTW include directory paths

The following variables will be checked by the function
  FFTW_USE_STATIC_LIBS       .. if true, only static libraries are searched
  FFTW_ROOT                  .. if set, search under this path first

Paths will be searched in the following order:
  FFTW_ROOT (if provided)
  PkgConfig paths (if found)
  Library/include installation directories
  Default find_* paths

The following component library locations will be defined (if found):
  FFTW_FLOAT_LIB
  FFTW_DOUBLE_LIB
  FFTW_LONGDOUBLE_LIB
  FFTW_FLOAT_THREADS_LIB
  FFTW_DOUBLE_THREADS_LIB
  FFTW_LONGDOUBLE_THREADS_LIB
  FFTW_FLOAT_OMP_LIB
  FFTW_DOUBLE_OMP_LIB
  FFTW_LONGDOUBLE_OMP_LIB

The following IMPORTED targets will be created (if found):
  FFTW::Float
  FFTW::Double
  FFTW::LongDouble
  FFTW::FloatThreads
  FFTW::DoubleThreads
  FFTW::LongDoubleThreads
  FFTW::FloatOMP
  FFTW::DoubleOMP
  FFTW::LongDoubleOMP
]]

include(FindPackageHandleStandardArgs)

if(NOT FFTW_ROOT AND DEFINED ENV{FFTWDIR})
    set(FFTW_ROOT $ENV{FFTWDIR})
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)

# Determine from PKG
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PKG_FFTW QUIET fftw3)
endif()

# Check whether to search static or dynamic libs
set(CMAKE_FIND_LIBRARY_SUFFIXES_SAV ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(${FFTW_USE_STATIC_LIBS})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV})
endif()

# Paths to pass to find_library for each component
set(findlib_paths
    ${FFTW_ROOT}
    ${PKG_FFTW_LIBRARY_DIRS}
    ${LIB_INSTALL_DIR}
    )

# Find include directory
find_path(FFTW_INCLUDE_DIRS
    NAMES fftw3.h
    PATHS ${FFTW_ROOT}
          ${PKG_FFTW_INCLUDE_DIRS}
          ${INCLUDE_INSTALL_DIR}
    PATH_SUFFIXES include
    )

set(FFTW_LIBRARIES "")

foreach(dtype Float Double LongDouble)

    # Single-letter suffix for the library name
    string(REGEX REPLACE "(.).*" "\\1" letter ${dtype})
    string(TOLOWER ${letter} letter)
    # The double-precision library doesn't use a suffix
    if("${letter}" STREQUAL "d")
        set(letter "")
    endif()

    foreach(system "" Threads OMP)

        # CamelCase component name used for interface libraries
        # e.g. FloatThreads
        set(component ${dtype}${system})

        # Component library location variable used via find_library
        # e.g. FFTW_DOUBLE_THREADS_LIB
        if(system)
            set(libvar FFTW_${dtype}_${system}_LIB)
        else()
            set(libvar FFTW_${dtype}_LIB)
        endif()
        string(TOUPPER ${libvar} libvar)

        # Filename root common to all libraries
        set(libname fftw3${letter})
        if(system)
            string(TOLOWER ${system} systemlower)
            set(libname ${libname}_${systemlower})
        endif()
        # Actual filenames looked for by find_library
        set(libnames
            ${libname}
            lib${libname}3-3
            )

        find_library(
            ${libvar}
            NAMES ${libnames}
            PATHS ${findlib_paths}
            PATH_SUFFIXES lib lib64
            )

        # Tell find_package whether this component was found
        set(FFTW_${component}_FIND_QUIETLY TRUE)
        find_package_handle_standard_args(FFTW_${component}
            HANDLE_COMPONENTS REQUIRED_VARS ${libvar} FFTW_INCLUDE_DIRS)
        # Also set the value of the legacy library-variable
        # (Will be set to *-NOTFOUND if not found)
        set(${libvar} ${FFTW_${component}})

        # If the library was found:
        if(${libvar} AND NOT TARGET FFTW::${component})
            # Add it to the list of FFTW libraries
            list(APPEND FFTW_LIBRARIES ${${libvar}})

            # Create a corresponding interface library
            add_library(FFTW::${component} IMPORTED INTERFACE)
            target_include_directories(
                FFTW::${component} SYSTEM INTERFACE ${FFTW_INCLUDE_DIRS})
            target_link_libraries(
                FFTW::${component} INTERFACE ${${libvar}})
        endif()

        mark_as_advanced(${libvar})

    endforeach()
endforeach()

# Restore saved find_library suffixes
set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_SAV})

find_package_handle_standard_args(FFTW
    REQUIRED_VARS FFTW_LIBRARIES FFTW_INCLUDE_DIRS
    HANDLE_COMPONENTS
    )

mark_as_advanced(
    FFTW_INCLUDE_DIRS
    FFTW_LIBRARIES
    )
