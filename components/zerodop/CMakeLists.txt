add_subdirectory(geo2rdr)
add_subdirectory(geozero)
add_subdirectory(topozero)

if(CMAKE_CUDA_COMPILER)
    # add_subdirectory(GPUampcor) TODO cublas_device removed from CUDA ≥ 10
    add_subdirectory(GPUgeo2rdr)
endif()

InstallSameDir(__init__.py)
