Python_add_library(offsetpoly MODULE
    bindings/offsetpolymodule.cpp
    src/offsetpolyState.F
    src/offsetpoly.f90
    src/offsetpolySetState.F
    src/offsetpolyAllocateDeallocate.F
    src/offsetpolyGetState.F
    )
target_include_directories(offsetpoly PUBLIC include)
target_link_libraries(offsetpoly PUBLIC
    combinedLib
    utilLib
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
