Python_add_library(geo2rdr MODULE
    bindings/geo2rdrmodule.cpp
    src/geo2rdrSetState.F
    src/geo2rdr.f90
    src/geo2rdrState.F
    )
target_include_directories(geo2rdr PUBLIC include)
target_link_libraries(geo2rdr PUBLIC
    DataAccessor_static
    combinedLib
    utilLib
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
