Python_add_library(mocompTSX MODULE
    bindings/mocompTSXmodule.cpp
    src/mocompTSXAllocateDeallocate.f
    src/mocompTSXGetState.f
    src/mocompTSXSetState.f
    src/mocompTSXState.f
    )
target_include_directories(mocompTSX PUBLIC include)
target_link_libraries(mocompTSX
    PUBLIC
    combinedLib
    formslcLib
    utilLib
    )

InstallSameDir(
    mocompTSX
    __init__.py
    MocompTSX.py
    )
