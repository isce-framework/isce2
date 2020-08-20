Python_add_library(simamplitude MODULE
    bindings/simamplitudemodule.cpp
    src/simamplitudeSetState.F
    src/simamplitude.f90
    src/simamplitudeState.F
    )
target_include_directories(simamplitude PRIVATE include)
target_link_libraries(simamplitude PRIVATE
    isce2::DataAccessorLib
    )
