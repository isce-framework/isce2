cython_add_module(GPUtopozero
    GPUtopozero.pyx
    cuda/gpuTopo.cu
    src/AkimaLib.cpp
    src/Ellipsoid.cpp
    src/LinAlg.cpp
    src/Orbit.cpp
    src/Peg.cpp
    src/PegTrans.cpp
    src/Poly2d.cpp
    src/Topo.cpp
    src/TopoMethods.cpp
    src/Topozero.cpp
    src/UniformInterp.cpp
    )
target_include_directories(GPUtopozero PRIVATE
    include
    )
target_compile_definitions(GPUtopozero PRIVATE
    GPU_ACC_ENABLED
    )
target_link_libraries(GPUtopozero PRIVATE
    isce2::DataAccessorLib
    OpenMP::OpenMP_CXX
    )
InstallSameDir(
    GPUtopozero
    __init__.py
    )
