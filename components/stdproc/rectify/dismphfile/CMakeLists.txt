InstallSameDir(
    __init__.py
    Dismphfile.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(dismphfile MODULE
    bindings/dismphfilemodule.cpp
    src/dismphfileSetState.F
    src/dismphfileState.F
    src/SConscript
    src/dismphfile.F
    src/writetiff.f
    )
target_include_directories(dismphfile PRIVATE include)
target_link_libraries(dismphfile PRIVATE
    isce2::DataAccessorLib
    )
InstallSameDir(dismphfile)
