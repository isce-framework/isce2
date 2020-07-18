# TODO (subdir-staticlib)
# This library is in a subdirectory but used by other libraries,
# so it's compiled as a static library so we don't need to worry
# about RPATH. Is there a better way to this?
add_library(formslcLib SHARED
    formslcLib/src/arraymodule.f90
    formslcLib/src/get_frate.f90
    formslcLib/src/io.c
    )
set_property(TARGET formslcLib PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(formslcLib PUBLIC
    utilLib
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

add_subdirectory(correct)
add_subdirectory(crossmul)
add_subdirectory(estamb)
add_subdirectory(formslc)
add_subdirectory(mocompTSX)
add_subdirectory(offsetpoly)
add_subdirectory(resamp)
add_subdirectory(resamp_amps)
add_subdirectory(resamp_image)
add_subdirectory(resamp_only)
add_subdirectory(resamp_slc)
add_subdirectory(topo)

InstallSameDir(__init__.py)
