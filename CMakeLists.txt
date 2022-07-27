cmake_minimum_required(VERSION 3.13...3.18)

project(isce2 LANGUAGES C CXX Fortran)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/.cmake)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    enable_language(CUDA)
    find_package(CUDAToolkit) # TODO added in cmake 3.17 - copy this module
endif()

find_package(Python 3.5 REQUIRED COMPONENTS Interpreter Development
                        OPTIONAL_COMPONENTS NumPy)
find_package(FFTW REQUIRED)
find_package(Motif)
find_package(OpenMP REQUIRED COMPONENTS C CXX Fortran)
find_package(OpenCV COMPONENTS core highgui imgproc)
find_package(pybind11 CONFIG)

# Find these, and create IMPORTED INTERFACE libraries for them if they exist
include(TargetGDAL)
include(TargetMotif)
include(TargetX11)
include(UseCython)

# If we're the root cmake project (e.g. not add_subdirectory):
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_LIST_DIR}")
    # override this to also test the resulting extension
    function(Python_add_library target)
        _Python_add_library(${target} ${ARGN})
        set(name "$<TARGET_PROPERTY:${target},OUTPUT_NAME>")
        add_test(NAME import_${target}
                 COMMAND ${Python_EXECUTABLE} -c
                     "import $<IF:$<BOOL:${name}>,${name},${target}>"
                 )
    endfunction()
endif()

if(NOT DEFINED PYTHON_MODULE_DIR)
    set(PYTHON_MODULE_DIR packages CACHE PATH "Python module directory")
endif()
if(NOT DEFINED ISCE2_PKG)
    set(ISCE2_PKG ${PYTHON_MODULE_DIR}/isce2 CACHE PATH
        "ISCE 2 python package install dir")
endif()

if(IS_ABSOLUTE "${ISCE2_PKG}")
    set(ISCE2_PKG_FULL "${ISCE2_PKG}")
else()
    set(ISCE2_PKG_FULL "${CMAKE_INSTALL_PREFIX}/${ISCE2_PKG}")
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

file(READ license.py LICENSE_TXT)
string(FIND "${LICENSE_TXT}" "stanford_license = None" match)
if(${match} EQUAL -1)
    set(ISCE2_HAVE_LICENSE YES)
else()
    set(ISCE2_HAVE_LICENSE NO)
endif()
option(ISCE2_WITH_STANFORD "Build Stanford components" ${ISCE2_HAVE_LICENSE})
if(ISCE2_WITH_STANFORD)
    InstallSameDir(license.py)
    message(STATUS "ISCE2's Stanford-licensed components will be built.")
else()
    message(STATUS "ISCE2's Stanford-licensed components will NOT be built.")
endif()

# We also need to create an empty directory for help
install(DIRECTORY DESTINATION ${ISCE2_PKG}/helper)

# CMake will install a python package named "isce2",
# but legacy scripts import it as simply "isce".
# Make a symlink isce -> isce2 for compatibility.
set(symsrc isce2)
if(IS_ABSOLUTE "${PYTHON_MODULE_DIR}")
    set(symdest "${PYTHON_MODULE_DIR}/isce")
else()
    set(symdest "${CMAKE_INSTALL_PREFIX}/${PYTHON_MODULE_DIR}/isce")
endif()
install(CODE "execute_process(COMMAND
    ${CMAKE_COMMAND} -E create_symlink ${symsrc} ${symdest})")

# Enable native packaging using CPack
if(NOT CPACK_PACKAGE_CONTACT)
    set(CPACK_PACKAGE_CONTACT "Ryan Burns <rtburns@jpl.nasa.gov>")
endif()
include(CPack)
