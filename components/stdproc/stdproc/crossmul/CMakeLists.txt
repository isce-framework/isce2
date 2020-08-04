Python_add_library(crossmul MODULE
    bindings/crossmulmodule.cpp
    src/crossmulState.F
    src/crossmul.f90
    )
target_include_directories(crossmul PUBLIC include)
if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(crossmul PUBLIC
        OpenMP::OpenMP_Fortran
        )
endif()
InstallSameDir(
    crossmul
    __init__.py
    Crossmul.py
    )
