isce2_add_staticlib(formslcSoiLib
    src/formslcAllocateDeallocate.F
    src/formslcGetState.F
    src/formslcSetState.F
    src/formslcStateSoi.f90
    )
target_include_directories(formslcSoiLib PUBLIC include)
target_link_libraries(formslcSoiLib PRIVATE
    isce2::utilLib
    isce2::formslcLib
    )
target_link_libraries(formslcSoiLib PUBLIC
    combinedLib
    )

Python_add_library(stdproc_formslc MODULE
    bindings/formslcmodule.cpp
    )
target_link_libraries(stdproc_formslc PUBLIC
    isce2::formslcSoiLib
    )

set_target_properties(stdproc_formslc
    PROPERTIES OUTPUT_NAME formslc)

InstallSameDir(
    stdproc_formslc
    __init__.py
    Formslc.py
    )
