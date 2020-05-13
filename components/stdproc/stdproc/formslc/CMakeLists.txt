Python_add_library(stdproc_formslc MODULE
    bindings/formslcmodule.cpp
    src/formslcAllocateDeallocate.F
    src/formslcGetState.F
    src/formslcSetState.F
    src/formslcStateSoi.f90
    )
target_include_directories(stdproc_formslc PUBLIC include)
target_link_libraries(stdproc_formslc PUBLIC
    combinedLib
    formslcLib
    utilLib
    )

set_target_properties(stdproc_formslc
    PROPERTIES OUTPUT_NAME formslc)

InstallSameDir(
    stdproc_formslc
    __init__.py
    Formslc.py
    )
