InstallSameDir(
    Getpeg.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(getpeg MODULE
    bindings/getpegmodule.cpp
    src/getpeg.F
    src/getpegAllocateDeallocate.F
    src/getpegGetState.F
    src/getpegSetState.F
    src/getpegState.F
    )
target_include_directories(getpeg PRIVATE include)
target_link_libraries(getpeg PRIVATE
    isce2::orbitLib
    isce2::stdoelLib
    )
InstallSameDir(
    getpeg
    )
