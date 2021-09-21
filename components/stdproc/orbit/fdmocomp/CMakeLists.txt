InstallSameDir(
    __init__.py
    Fdmocomp.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(fdmocomp MODULE
    bindings/fdmocompmodule.cpp
    src/fdmocomp.f90
    src/fdmocompAllocateDeallocate.f
    src/fdmocompGetState.f
    src/fdmocompSetState.f
    src/fdmocompState.f
    )
target_include_directories(fdmocomp PRIVATE include)
target_link_libraries(fdmocomp PRIVATE
    isce2::utilLib
    )
InstallSameDir(
    fdmocomp
    )
