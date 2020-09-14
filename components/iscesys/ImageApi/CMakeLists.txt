isce2_add_staticlib(DataAccessorLib
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
target_include_directories(DataAccessorLib PUBLIC
    DataAccessor/include
    DataCaster/include
    Factories/include
    InterleavedAccessor/include
    )
target_link_libraries(DataAccessorLib PUBLIC
    isce2::combinedLib
    )

if(TARGET GDAL::GDAL)
    target_sources(DataAccessorLib PRIVATE
        InterleavedAccessor/src/GDALAccessor.cpp
        )
    target_link_libraries(DataAccessorLib PUBLIC
        GDAL::GDAL
        )
else()
    target_compile_definitions(DataAccessorLib PRIVATE -DHAVE_GDAL=0)
endif()

Python_add_library(DataAccessor MODULE
    DataAccessor/bindings/DataAccessormodule.cpp
    )
target_link_libraries(DataAccessor PRIVATE isce2::DataAccessorLib)

InstallSameDir(
    Factories/CasterFactory.py
    DataAccessor/DataAccessorPy.py
    DataAccessor
    )
