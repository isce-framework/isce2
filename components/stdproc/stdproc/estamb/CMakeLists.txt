Python_add_library(estamb MODULE
    bindings/estambmodule.cpp
    src/estambAllocateDeallocate.F
    src/estambGetState.F
    src/estambSetState.F
    src/estambStateSoi.f90
    )
target_include_directories(estamb PUBLIC include)

InstallSameDir(
    estamb
    __init__.py
    Estamb.py
    )
