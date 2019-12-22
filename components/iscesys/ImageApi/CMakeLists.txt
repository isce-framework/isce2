# TODO (subdir-staticlib)
# This library is in a subdirectory but used by other libraries,
# so it's compiled as a static library so we don't need to worry
# about RPATH. Is there a better way to this?
add_library(DataAccessor_static STATIC
    DataAccessor/src/DataAccessorCaster.cpp
    DataAccessor/src/DataAccessor.cpp
    DataAccessor/src/DataAccessorF.cpp
    DataAccessor/src/DataAccessorNoCaster.cpp
    Factories/src/AccessorFactory.cpp
    Factories/src/CasterFactory.cpp
    Factories/src/InterleavedFactory.cpp
    InterleavedAccessor/src/BILAccessor.cpp
    InterleavedAccessor/src/BIPAccessor.cpp
    InterleavedAccessor/src/BSQAccessor.cpp
    InterleavedAccessor/src/InterleavedAccessor.cpp
    InterleavedAccessor/src/InterleavedBase.cpp
    InterleavedAccessor/src/Poly1dInterpolator.cpp
    InterleavedAccessor/src/Poly2dInterpolator.cpp
    )

set_property(TARGET DataAccessor_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(DataAccessor_static PUBLIC
    DataAccessor/include
    DataCaster/include
    Factories/include
    InterleavedAccessor/include
    )
target_link_libraries(DataAccessor_static PUBLIC
    combinedLib
    )

if(TARGET GDAL::GDAL)
    target_sources(DataAccessor_static PRIVATE
        InterleavedAccessor/src/GDALAccessor.cpp
        )
    target_link_libraries(DataAccessor_static PUBLIC
        GDAL::GDAL
        )
else()
    target_compile_definitions(DataAccessor_static PRIVATE -DHAVE_GDAL=0)
endif()

Python_add_library(DataAccessor MODULE
    DataAccessor/bindings/DataAccessormodule.cpp
    )
target_link_libraries(DataAccessor PRIVATE DataAccessor_static)

InstallSameDir(
    Factories/CasterFactory.py
    DataAccessor/DataAccessorPy.py
    DataAccessor
    )
