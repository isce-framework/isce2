Python_add_library(geo2rdr MODULE
    bindings/geo2rdrmodule.cpp
    src/geo2rdrSetState.F
    src/geo2rdr.f90
    src/geo2rdrState.F
    )
target_include_directories(geo2rdr PUBLIC include)
target_link_libraries(geo2rdr PUBLIC
    isce2::DataAccessorLib
    isce2::combinedLib
    isce2::utilLib
    )

if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(geo2rdr PRIVATE
        OpenMP::OpenMP_Fortran
        )
endif()

InstallSameDir(
    geo2rdr
    __init__.py
    Geo2rdr.py
    )
