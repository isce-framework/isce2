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
    utilLib
    )

InstallSameDir(
    resamp
    __init__.py
    Resamp.py
    )
