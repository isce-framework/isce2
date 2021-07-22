InstallSameDir(
    __init__.py
    Correct.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(correct MODULE
    bindings/correctmodule.cpp
    src/correct.f
    src/correctAllocateDeallocate.f
    src/correctSetState.f
    src/correctState.f
    )
target_include_directories(correct PRIVATE include)
target_link_libraries(correct PRIVATE
    isce2::DataAccessorLib
    isce2::orbitLib
    isce2::utilLib
    OpenMP::OpenMP_Fortran
    )
