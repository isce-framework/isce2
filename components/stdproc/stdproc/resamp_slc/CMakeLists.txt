Python_add_library(resamp_slc MODULE
    bindings/resamp_slcmodule.cpp
    src/resamp_slc.f90
    src/resamp_slcSetState.F
    src/resamp_slcMethods.f
    src/resamp_slcState.F
    )
target_include_directories(resamp_slc PUBLIC include)
target_link_libraries(resamp_slc PUBLIC
    combinedLib
    utilLib
    )
target_compile_options(resamp_slc PRIVATE
    -ffree-line-length-none
    )
if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(resamp_slc PUBLIC
        OpenMP::OpenMP_Fortran
        )
endif()

InstallSameDir(
    resamp_slc
    __init__.py
    Resamp_slc.py
    )
