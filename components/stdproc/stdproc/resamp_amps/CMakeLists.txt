Python_add_library(resamp_amps MODULE
    bindings/resamp_ampsmodule.cpp
    src/resamp_amps.f90
    src/resamp_ampsAllocateDeallocate.F
    src/resamp_ampsGetState.F
    src/resamp_ampsSetState.F
    src/resamp_ampsState.F
    )
target_include_directories(resamp_amps PUBLIC include)
target_link_libraries(resamp_amps PRIVATE
    isce2::DataAccessorLib
    isce2::resampLib
    )

InstallSameDir(
    resamp_amps
    __init__.py
    Resamp_amps.py
    )
