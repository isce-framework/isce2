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
target_include_directories(formslcLib PRIVATE formslcLib/include)
target_link_libraries(formslcLib PUBLIC
    utilLib
    )
# TODO (fortran module include)
# This seems to be needed to use this library's modules,
# but is there a more idiomatic way to do this?
target_include_directories(formslcLib INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}
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
