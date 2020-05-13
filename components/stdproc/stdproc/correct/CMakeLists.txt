Python_add_library(correct MODULE
    bindings/correctmodule.cpp
    src/correctAllocateDeallocate.f
    src/correctSetState.f
    src/correctState.f
    )
target_include_directories(correct PUBLIC include)

InstallSameDir(
    correct
    __init__.py
    Correct.py
    )
