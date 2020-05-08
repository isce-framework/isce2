Python_add_library(ampcor MODULE
    bindings/ampcormodule.cpp
    src/ampcor.F
    src/ampcorAllocateDeallocate.F
    src/ampcorGetState.F
    src/ampcorPrintState.F
    src/ampcorSetState.F
    src/ampcorState.F
    )
target_include_directories(ampcor PUBLIC include)
target_link_libraries(ampcor PUBLIC
    utilLib
    DataAccessor_static
    )

InstallSameDir(
    ampcor
    __init__.py
    Ampcor.py
    DenseAmpcor.py
    NStage.py
    )
