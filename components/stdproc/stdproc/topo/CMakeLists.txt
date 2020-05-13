Python_add_library(topo MODULE
    bindings/topomodule.cpp
    src/topoAllocateDeallocate.f
    src/topoGetState.f
    src/topoMethods.f
    src/topoSetState.f
    src/topoState.f
    )
target_include_directories(topo PUBLIC include)
target_link_libraries(topo PUBLIC
    combinedLib
    utilLib
    )

InstallSameDir(
    topo
    __init__.py
    Topo.py
    )
