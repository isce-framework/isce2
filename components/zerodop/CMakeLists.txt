add_subdirectory(geo2rdr)
add_subdirectory(geozero)
add_subdirectory(topozero)

if(CMAKE_CUDA_COMPILER)

    # cublas_device removed from CUDA ≥ 10
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
       CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10)
        add_subdirectory(GPUampcor)
    endif()

    add_subdirectory(GPUgeo2rdr)
endif()

InstallSameDir(__init__.py)
