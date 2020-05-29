# Early exit if prereqs not available
if(NOT CMAKE_CUDA_COMPILER OR
   NOT TARGET GDAL::GDAL OR
   NOT TARGET Python::NumPy)
    return()
endif()

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

cython_add_module(PyCuAmpcor
    src/PyCuAmpcor.pyx
    src/GDALImage.cu
    src/SConscript
    src/SlcImage.cu
    src/cuAmpcorChunk.cu
    src/cuAmpcorController.cu
    src/cuAmpcorParameter.cu
    src/cuArrays.cu
    src/cuArraysCopy.cu
    src/cuArraysPadding.cu
    src/cuCorrFrequency.cu
    src/cuCorrNormalization.cu
    src/cuCorrTimeDomain.cu
    src/cuDeramp.cu
    src/cuEstimateStats.cu
    src/cuOffset.cu
    src/cuOverSampler.cu
    src/cuSincOverSampler.cu
    )
target_include_directories(PyCuAmpcor PRIVATE
    src
    ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES} # <cuda_runtime.h>
    )
target_link_directories(PyCuAmpcor PRIVATE
    ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} # cufft
    )
target_link_libraries(PyCuAmpcor PRIVATE
    GDAL::GDAL
    Python::NumPy
    cufft
    )

InstallSameDir(
    __init__.py
    PyCuAmpcor
    )
