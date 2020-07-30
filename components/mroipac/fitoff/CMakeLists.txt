Python_add_library(fitoff MODULE
    bindings/fitoffmodule.cpp
    src/fitoffGetState.F
    src/fitoffAllocateDeallocate.F
    src/fitoffSetState.F
    src/fitoff.F
    src/fitoffState.F
    )
target_include_directories(fitoff PUBLIC include)
target_link_libraries(fitoff PUBLIC
    isce2::combinedLib
    isce2::utilLib
    )

InstallSameDir(
    fitoff
    __init__.py
    Fitoff.py
    )
