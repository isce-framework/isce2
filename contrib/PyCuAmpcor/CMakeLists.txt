# Early exit if prereqs not available
if(NOT TARGET GDAL::GDAL
OR NOT TARGET Python::NumPy
OR NOT TARGET CUDA::cublas
OR NOT TARGET CUDA::cufft
   )
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
    )
target_link_libraries(PyCuAmpcor PRIVATE
    CUDA::cufft
    CUDA::cublas
    GDAL::GDAL
    Python::NumPy
    )

InstallSameDir(
    __init__.py
    PyCuAmpcor
    )
