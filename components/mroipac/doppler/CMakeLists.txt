Python_add_library(doppler MODULE
    bindings/dopplermodule.cpp
    src/doppler.f
    src/dopplerAllocateDeallocate.f
    src/dopplerGetState.f
    src/dopplerSetState.f
    src/dopplerState.f
    )
target_include_directories(doppler PUBLIC include)
target_link_libraries(doppler PUBLIC DataAccessor_static)

InstallSameDir(
    doppler
    __init__.py
    Doppler.py
    )
