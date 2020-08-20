add_subdirectory(ImageUtil)
add_subdirectory(geo)
add_subdirectory(offoutliers)

Python_add_library(offoutliers MODULE
    offoutliers/bindings/offoutliersmodule.cpp
    )
target_link_libraries(offoutliers PRIVATE isce2::offoutliersLib)

add_definitions(-DHAVE_CONFIG_H -DHAVE_FFTW=1)
add_subdirectory(src)
target_include_directories(utilLib PUBLIC
    include
    )
target_link_libraries(utilLib PRIVATE
    FFTW::Float
    )

isce2_add_staticlib(combinedLib
    Library/geometry/src/geometryModule.F
    Library/linalg3/src/linalg3Module.F
    Library/linalg3/src/linalg3.c
    Library/orbit/src/orbit.c
    Library/orbit/src/orbitModule.F
    Library/orbit/src/orbitHermite.c
    Library/poly1d/src/poly1d.c
    Library/poly1d/src/poly1dModule.F
    Library/poly2d/src/poly2d.c
    Library/poly2d/src/poly2dModule.F
    )
target_include_directories(combinedLib PUBLIC
    Library/geometry/include
    Library/linalg3/include
    Library/orbit/include
    Library/poly1d/include
    Library/poly2d/include
    )

Python_add_library(combinedlibmodule MODULE
    Library/bindings/combinedlibmodule.cpp
    )
target_include_directories(combinedlibmodule PUBLIC
    Library/include
    )
target_link_libraries(combinedlibmodule PUBLIC
    isce2::combinedLib
    )

# Set up fortran module paths
set(mdir ${CMAKE_CURRENT_BINARY_DIR}/utillib_fortran_modules)
set_property(TARGET utilLib PROPERTY Fortran_MODULE_DIRECTORY ${mdir})
target_include_directories(utilLib INTERFACE
    $<$<COMPILE_LANGUAGE:Fortran>:${mdir}>
    )
set(mdir ${CMAKE_CURRENT_BINARY_DIR}/combinelib_fortran_modules)
set_property(TARGET combinedLib PROPERTY Fortran_MODULE_DIRECTORY ${mdir})
target_include_directories(combinedLib INTERFACE
    $<$<COMPILE_LANGUAGE:Fortran>:${mdir}>
    )

add_subdirectory(simamplitude)

InstallSameDir(
    simamplitude
    combinedlibmodule
    offoutliers
    __init__.py
    decorators.py
    mathModule.py
    offoutliers/Offoutliers.py
    py2to3.py
    StringUtils.py
    Library/python/Poly1D.py
    Library/python/Poly2D.py
    Library/python/PolyFactory.py
    Library/python/Polynomial.py
    simamplitude/Simamplitude.py
    )
