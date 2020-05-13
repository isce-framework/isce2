cython_add_module(GPUampcor
    GPUampcor.pyx
    cuda/GPUamp.cu
    src/Ampcor.cpp
    src/AmpcorFFT.cpp
    src/AmpcorMethods.cpp
    )
target_include_directories(GPUampcor PUBLIC
    include
    )
target_link_libraries(GPUampcor PRIVATE
    cublas
    DataAccessor_static
    )
