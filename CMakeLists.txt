cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(isce2 LANGUAGES C CXX Fortran)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/.cmake)

find_package(Python 3.5 REQUIRED COMPONENTS Interpreter Development)
find_package(FFTW)

if(NOT TARGET FFTW::Float)
    message(STATUS "FFTW not found. Falling back to Intel MKL wrapper...")

    # Look for suitable MKL runtime
    # (Move this to a separate FindMKL helper if it gets too complicated.)
    find_library(mkl_lib NAMES mkl_rt)
    if(mkl_lib)
        message(STATUS "Found MKL library: ${mkl_lib}")
    else()
        message(FATAL_ERROR "Could not find FFTW or MKL library")
    endif()

    # Create a wrapper library to mimic the FFTW equivalent
    add_library(FFTW::Float IMPORTED INTERFACE)
    target_include_directories(FFTW::Float INTERFACE extern/mkl/include)
    target_link_libraries(FFTW::Float INTERFACE ${mkl_lib})
endif()

find_package(Motif)
find_package(OpenMP REQUIRED COMPONENTS C CXX Fortran)
# Find these, and create IMPORTED INTERFACE libraries for them if they exist
include(TargetGDAL)
include(TargetMotif)
include(TargetX11)
include(UseCython)

if(NOT DEFINED PYTHON_MODULE_DIR)
    set(PYTHON_MODULE_DIR packages CACHE PATH
        "Python module directory (relative to install prefix)")
endif()
if(NOT DEFINED ISCE2_PKG)
    set(ISCE2_PKG ${PYTHON_MODULE_DIR}/isce2 CACHE PATH
        "ISCE 2 python package install dir (relative to install prefix)")
endif()

include(isce2_buildflags)
include(isce2_helpers)

enable_testing()

add_subdirectory(applications)
add_subdirectory(components)
add_subdirectory(contrib components/contrib)
add_subdirectory(defaults)
add_subdirectory(library)
add_subdirectory(test)

InstallSameDir(
    __init__.py
    release_history.py
    )

# We also need to create an empty directory for help
install(DIRECTORY DESTINATION ${ISCE2_PKG}/helper)

# CMake will install a python package named "isce2",
# but legacy scripts import it as simply "isce".
# Make a symlink isce -> isce2 for compatibility.
set(symsrc ${CMAKE_INSTALL_PREFIX}/${ISCE2_PKG})
set(symdest ${symsrc}/../isce)
install(CODE "execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink ${symsrc} ${symdest})")
