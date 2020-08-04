Python_add_library(get_peg_info MODULE
    bindings/get_peg_infomodule.cpp
    src/get_peg_info.F
    src/get_peg_infoSetState.F
    src/get_peg_infoAllocateDeallocate.F
    src/get_peg_infoGetState.F
    src/get_peg_infoState.F
    )
target_include_directories(get_peg_info PUBLIC include)

InstallSameDir(
    get_peg_info
    __init__.py
    Get_peg_info.py
    )
