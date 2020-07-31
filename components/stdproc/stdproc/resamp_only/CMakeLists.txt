Python_add_library(resamp_only MODULE
    bindings/resamp_onlymodule.cpp
    src/resamp_onlySetState.F
    src/resamp_only.f90
    src/resamp_onlyAllocateDeallocate.F
    src/resamp_onlyGetState.F
    src/resamp_onlyState.F
    )
target_include_directories(resamp_only PUBLIC include)
target_link_libraries(resamp_only PRIVATE
    isce2::DataAccessorLib
    isce2::resampLib
    isce2::stdoelLib
    )

InstallSameDir(
    resamp_only
    __init__.py
    Resamp_only.py
    )
