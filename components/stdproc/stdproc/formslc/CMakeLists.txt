InstallSameDir(
    __init__.py
    Formslc.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(stdproc_formslc MODULE
    bindings/formslcmodule.cpp
    src/formslc.f90
    src/formslcAllocateDeallocate.F
    src/formslcGetState.F
    src/formslcSetState.F
    src/formslcStateSoi.f90
    )
target_include_directories(stdproc_formslc PRIVATE include)
target_link_libraries(stdproc_formslc PRIVATE
    isce2::DataAccessorLib
    isce2::formslcLib
    )
set_target_properties(stdproc_formslc
    PROPERTIES OUTPUT_NAME formslc
    )
InstallSameDir(
    stdproc_formslc
    )
