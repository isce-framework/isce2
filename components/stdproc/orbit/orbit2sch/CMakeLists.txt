InstallSameDir(
    Orbit2sch.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(orbit2sch MODULE
    bindings/orbit2schmodule.cpp
    src/orbit2sch.F
    src/orbit2schAllocateDeallocate.F
    src/orbit2schGetState.F
    src/orbit2schSetState.F
    src/orbit2schState.F
    )
target_include_directories(orbit2sch PRIVATE include)
target_link_libraries(orbit2sch PRIVATE
    isce2::orbitLib
    isce2::stdoelLib
    isce2::utilLib
    )
InstallSameDir(
    orbit2sch
    )
