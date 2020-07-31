Python_add_library(doppler MODULE
    bindings/dopplermodule.cpp
    src/doppler.f
    src/dopplerAllocateDeallocate.f
    src/dopplerGetState.f
    src/dopplerSetState.f
    src/dopplerState.f
    )
target_include_directories(doppler PUBLIC include)
target_link_libraries(doppler PUBLIC isce2::DataAccessorLib)

InstallSameDir(
    doppler
    __init__.py
    Doppler.py
    )
