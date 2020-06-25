add_subdirectory(ImageUtil)
add_subdirectory(geo)

Python_add_library(offoutliers MODULE
    offoutliers/bindings/offoutliersmodule.cpp
    offoutliers/src/offoutliers.F
    offoutliers/src/offoutliersAllocateDeallocate.F
    offoutliers/src/offoutliersGetState.F
    offoutliers/src/offoutliersSetState.F
    offoutliers/src/offoutliersState.F
    )
target_include_directories(offoutliers PUBLIC offoutliers/include)
target_link_libraries(offoutliers PUBLIC
    stdoel_static
    )

add_definitions(-DHAVE_CONFIG_H -DHAVE_FFTW=1)

add_library(utilLib SHARED
    src/akima_reg.F
    src/args_roi.F
    src/besseldiffs.F
    src/bilinear.F
    src/cfft1d_jpl.F
    src/cfft2d.F
    src/cffts.F
    src/config.h
    src/convert_schdot_to_xyzdot.F
    src/convert_sch_to_xyz.F
    src/cross.F
    src/curvature.F
    src/derampc.F
    src/dop.F
    src/dot.F
    src/enubasis.F
    src/fc.F
    src/fc.f.org
    src/fftw3stub.c
    src/fftw3stub.cc
    src/fortranUtils.f90
    src/fourn.F
    src/fournnr.F
    src/getangs.F
    src/gettcn_tcvec.F
    src/hunt.F
    src/inter_motion.F
    src/interp.F
    src/intp_coef.f90
    src/intpcoefnorm.F
    src/io.c
    src/latlon.F
    src/latlon_nostruct.F
    #src/lfit.F
    src/linalg.f90
    src/lincomb.F
    src/lookvec.F
    src/lsq.f90
    src/matmat.F
    src/matvec.F
    src/norm.F
    src/orrmread1.F
    src/polint.F
    src/PowerOfTwo.cc
    src/quadfit.f90
    src/radar_to_xyz.F
    src/rdf_common.inc
    src/roi_exit.cc
    src/schbasis.F
    src/second.c
    src/sfftw_import.c
    src/spline.f
    #src/svd.F
    #src/svdvecfit9.F
    #src/svdvecfit.F
    src/tranmat.F
    src/uniform_interp.f90
    src/unitvec.F
    src/utmtoll.F
    src/zbrent.F
    )
target_include_directories(utilLib PUBLIC
    include
    )
target_link_libraries(utilLib PRIVATE
    FFTW::Float
    )

add_library(combinedLib SHARED
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
    combinedLib
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

install(TARGETS
    utilLib
    combinedLib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

InstallSameDir(
    combinedlibmodule
    offoutliers
    __init__.py
    decorators.py
    mathModule.py
    offoutliers/Offoutliers.py
    StringUtils.py
    Library/python/Poly1D.py
    Library/python/Poly2D.py
    Library/python/PolyFactory.py
    Library/python/Polynomial.py
    )
