Python_add_library(dopiq MODULE
    bindings/dopiqmodule.cpp
    src/dopiq-new.f
    src/dopiqAllocateDeallocate.f
    src/dopiqGetState.f
    src/dopiqSetState.f
    src/dopiqState.f
    )
target_include_directories(dopiq PUBLIC include)
target_link_libraries(dopiq PUBLIC DataAccessor_static)

InstallSameDir(
    dopiq
    __init__.py
    DopIQ.py
    )
