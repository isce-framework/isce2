cython_add_module(GPUgeo2rdr
    GPUgeo2rdr.pyx
    cuda/GPUgeo.cu
    src/Ellipsoid.cpp
    src/Geo2rdr.cpp
    src/GeoController.cpp
    src/LinAlg.cpp
    src/Orbit.cpp
    src/Poly1d.cpp
    )
target_include_directories(GPUgeo2rdr PUBLIC
    include
    )
target_link_libraries(GPUgeo2rdr PRIVATE
    DataAccessor_static
    )
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(GPUgeo2rdr PRIVATE OpenMP::OpenMP_CXX)
endif()
