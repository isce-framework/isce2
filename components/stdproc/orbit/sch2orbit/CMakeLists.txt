InstallSameDir(
    Sch2orbit.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(sch2orbit MODULE
    bindings/sch2orbitmodule.cpp
    src/sch2orbit.F
    src/sch2orbitAllocateDeallocate.F
    src/sch2orbitGetState.F
    src/sch2orbitSetState.F
    src/sch2orbitState.F
    )
target_include_directories(sch2orbit PRIVATE include)
target_link_libraries(sch2orbit PRIVATE
    isce2::orbitLib
    isce2::stdoelLib
    isce2::utilLib
    )
InstallSameDir(
    sch2orbit
    )

