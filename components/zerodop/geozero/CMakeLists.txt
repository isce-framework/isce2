Python_add_library(geozero MODULE
    bindings/geozeromodule.cpp
    src/geozero.f90
    src/geozeroGetState.F
    src/geozeroMethods.F
    src/geozeroReadWrite.F
    src/geozeroSetState.F
    src/geozeroState.F
    src/SConscript
    )
target_include_directories(geozero PUBLIC include)
target_link_libraries(geozero PUBLIC
    isce2::DataAccessorLib
    isce2::combinedLib
    isce2::utilLib
    OpenMP::OpenMP_Fortran
    )

InstallSameDir(
    geozero
    __init__.py
    Geozero.py
    Geocodable.py
    )
