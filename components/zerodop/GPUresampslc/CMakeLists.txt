cython_add_module(GPUresampslc
    GPUresampslc.pyx
    cuda/GPUresamp.cu
    src/Interpolator.cpp
    src/Poly2d.cpp
    src/ResampMethods.cpp
    src/ResampSlc.cpp
    )
target_include_directories(GPUresampslc PRIVATE
    include
    )
target_compile_definitions(GPUresampslc PRIVATE
    GPU_ACC_ENABLED
    )
target_link_libraries(GPUresampslc PRIVATE
    isce2::DataAccessorLib
    OpenMP::OpenMP_CXX
    )
InstallSameDir(
    GPUresampslc
    __init__.py
    )
