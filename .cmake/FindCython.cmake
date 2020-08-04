# Tries to run Cython using `python -m cython`
execute_process(COMMAND ${Python_EXECUTABLE} -m cython --help
                RESULT_VARIABLE cython_status
                ERROR_QUIET OUTPUT_QUIET)

if(NOT cython_status)
    set(CYTHON_EXECUTABLE ${Python_EXECUTABLE} -m cython CACHE STRING
        "Cython executable")
endif()

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Cython REQUIRED_VARS CYTHON_EXECUTABLE)

mark_as_advanced(CYTHON_EXECUTABLE)
