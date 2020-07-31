Python_add_library(topozero MODULE
    bindings/topozeromodule.cpp
    src/topozero.f90
    src/topozeroGetState.f
    src/topozeroMethods.f
    src/topozeroSetState.f
    src/topozeroState.f
    )
target_include_directories(topozero PUBLIC include)
target_link_libraries(topozero PRIVATE
    isce2::combinedLib
    isce2::utilLib
    isce2::DataAccessorLib
    )
set_source_files_properties(src/topozero.f90 PROPERTIES COMPILE_OPTIONS -cpp)
if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(topozero PUBLIC
        OpenMP::OpenMP_Fortran
        )
endif()

InstallSameDir(
    topozero
    __init__.py
    Topozero.py
    )
