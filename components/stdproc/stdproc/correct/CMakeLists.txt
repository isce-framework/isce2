Python_add_library(correct MODULE
    bindings/correctmodule.cpp
    src/correctAllocateDeallocate.f
    src/correctSetState.f
    src/correctState.f
    )
target_include_directories(correct PUBLIC include)
if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(correct PUBLIC
        OpenMP::OpenMP_Fortran
        )
endif()
InstallSameDir(
    correct
    __init__.py
    Correct.py
    )
