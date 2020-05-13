Python_add_library(crossmul MODULE
    bindings/crossmulmodule.cpp
    src/crossmulState.F
    src/crossmul.f90
    )
target_include_directories(crossmul PUBLIC include)

InstallSameDir(
    crossmul
    __init__.py
    Crossmul.py
    )
