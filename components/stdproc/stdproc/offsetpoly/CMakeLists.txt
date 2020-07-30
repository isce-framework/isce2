isce2_add_staticlib(offsetpolyLib
    src/offsetpolyState.F
    src/offsetpoly.f90
    src/offsetpolySetState.F
    src/offsetpolyAllocateDeallocate.F
    src/offsetpolyGetState.F
    )
target_link_libraries(offsetpolyLib PRIVATE
    utilLib
    )

Python_add_library(offsetpoly MODULE
    bindings/offsetpolymodule.cpp
    )
target_include_directories(offsetpoly PUBLIC include)
target_link_libraries(offsetpoly PRIVATE
    isce2::offsetpolyLib
    isce2::resampLib
    stdoel_static
    )
if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(offsetpoly PUBLIC
        OpenMP::OpenMP_Fortran
        )
endif()
InstallSameDir(
    offsetpoly
    __init__.py
    Offsetpoly.py
    )
