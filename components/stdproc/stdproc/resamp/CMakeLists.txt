Python_add_library(resamp MODULE
    bindings/resampmodule.cpp
    src/resamp.f90
    src/resampAllocateDeallocate.F
    src/resampGetState.F
    src/resampSetState.F
    src/resampState.F
    )
target_include_directories(resamp PUBLIC include)
target_link_libraries(resamp PUBLIC
    isce2::resampLib
    isce2::utilLib
    isce2::stdoelLib
    isce2::DataAccessorLib
    )
if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(resamp PUBLIC
        OpenMP::OpenMP_Fortran
        )
endif()
InstallSameDir(
    resamp
    __init__.py
    Resamp.py
    )
