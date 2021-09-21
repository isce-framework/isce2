InstallSameDir(
    __init__.py
    Mocompbaseline.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(mocompbaseline MODULE
    bindings/mocompbaselinemodule.cpp
    src/mocompbaselineSetState.F
    src/mocompbaselineGetState.F
    src/mocompbaselineState.F
    src/mocompbaseline.f90
    src/mocompbaselineAllocateDeallocate.F
    )
target_include_directories(mocompbaseline PRIVATE include)
target_link_libraries(mocompbaseline PRIVATE
    isce2::orbitLib
    isce2::stdoelLib
    )
InstallSameDir(
    mocompbaseline
    )
