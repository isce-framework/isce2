InstallSameDir(
    __init__.py
    MocompTSX.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(mocompTSX MODULE
    bindings/mocompTSXmodule.cpp
    src/mocompTSXAllocateDeallocate.f
    src/mocompTSXState.f
    src/mocompTSX.f90
    src/mocompTSXGetState.f
    src/mocompTSXSetState.f
    )
target_include_directories(mocompTSX PRIVATE include)
target_link_libraries(mocompTSX PRIVATE
    isce2::formslcLib
    isce2::DataAccessorLib
    )
InstallSameDir(
    mocompTSX
    )
