isce2_add_staticlib(formslcLib
    src/arraymodule.f90
    src/get_frate.f90
    src/io.c
    )
target_link_libraries(formslcLib PUBLIC
    isce2::utilLib
    )
if(TARGET OpenMP::OpenMP_Fortran)
    target_link_libraries(formslcLib PUBLIC
        OpenMP::OpenMP_Fortran
        )
endif()
set(mdir ${CMAKE_CURRENT_BINARY_DIR}/formslc_fortran_modules)
set_property(TARGET formslcLib PROPERTY Fortran_MODULE_DIRECTORY ${mdir})
target_include_directories(formslcLib INTERFACE
    $<$<COMPILE_LANGUAGE:Fortran>:${mdir}>
    )
