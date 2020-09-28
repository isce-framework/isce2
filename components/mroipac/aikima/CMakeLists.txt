Python_add_library(aikima MODULE
    bindings/aikimamodule.cpp
    src/aikima.f90
    src/aikimaLib.F
    src/aikimaSetState.F
    src/aikimaState.F
    )
target_include_directories(aikima PUBLIC include)
target_link_libraries(aikima PUBLIC
    isce2::DataAccessorLib
    )

InstallSameDir(
    aikima
    __init__.py
    Aikima.py
    )
