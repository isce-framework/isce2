# Early exit if prereqs not available
if(NOT TARGET GDAL::GDAL
OR NOT TARGET CUDA::cufft
OR NOT pybind11_FOUND
   )
    return()
endif()

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

pybind11_add_module(PyCuAmpcor
    src/PyCuAmpcor.cpp
    src/GDALImage.cu
    src/SConscript
    src/cuAmpcorChunk.cu
    src/cuAmpcorController.cu
    src/cuAmpcorParameter.cu
    src/cuArrays.cu
    src/cuArraysCopy.cu
    src/cuArraysPadding.cu
    src/cuCorrFrequency.cu
    src/cuCorrNormalization.cu
    src/cuCorrNormalizationSAT.cu
    src/cuCorrNormalizer.cu
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
    GDAL::GDAL
    )

InstallSameDir(
    __init__.py
    PyCuAmpcor
    )
