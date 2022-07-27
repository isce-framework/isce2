InstallSameDir(
    Setmocomppath.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(setmocomppath MODULE
    bindings/setmocomppathmodule.cpp
    src/setmocomppath.F
    src/setmocomppathAllocateDeallocate.F
    src/setmocomppathState.F
    src/setmocomppathSetState.F
    src/setmocomppathGetState.F
    )
target_include_directories(setmocomppath PRIVATE include)
target_link_libraries(setmocomppath PRIVATE
    isce2::orbitLib
    isce2::stdoelLib
    )
InstallSameDir(
    setmocomppath
    )
