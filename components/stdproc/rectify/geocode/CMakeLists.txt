Python_add_library(geocode MODULE
    bindings/geocodemodule.cpp
    src/coordinates.f90
    src/geocode.f90
    src/geocodeAllocateDeallocate.F
    src/geocodeGetState.F
    src/geocodeMethods.F
    src/geocodeReadWrite.F
    src/geocodeSetState.F
    src/geocodeState.F
    )
target_include_directories(geocode PUBLIC include)
target_link_libraries(geocode PUBLIC
    DataAccessor_static
    OpenMP::OpenMP_CXX
    stdoel_static
    combinedLib
    utilLib
    )

InstallSameDir(
    geocode
    __init__.py
    Geocode.py
    Geocodable.py
    )
