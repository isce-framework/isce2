InstallSameDir(
    __init__.py
    Estamb.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

Python_add_library(estamb MODULE
    bindings/estambmodule.cpp
    src/estamb.f90
    src/estambAllocateDeallocate.F
    src/estambGetState.F
    src/estambSetState.F
    src/estambStateSoi.f90
    )
target_include_directories(estamb PRIVATE include)
target_link_libraries(estamb PRIVATE
    isce2::DataAccessorLib
    isce2::formslcLib
    )
