if(NOT TARGET CUDA::cublas)
    return()
endif()

return() # TODO get cublas_device working or remove usage of it

cython_add_module(GPUampcor
    GPUampcor.pyx
    cuda/GPUamp.cu
    src/Ampcor.cpp
    src/AmpcorFFT.cpp
    src/AmpcorMethods.cpp
    )
target_include_directories(GPUampcor PRIVATE
    include
    )
target_compile_definitions(GPUampcor PRIVATE
    GPU_ACC_ENABLED
    )
target_link_libraries(GPUampcor PRIVATE
    CUDA::cublas
    isce2::DataAccessorLib
    FFTW::Float
    )
InstallSameDir(
    GPUampcor
    __init__.py
    )
