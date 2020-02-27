InstallSameDir(__init__.py)

if(CYTHON_EXECUTABLE AND TARGET GDAL::GDAL)
    cython_add_module(splitSpectrum
        pyx/splitRangeSpectrum.pyx
        src/splitRangeSpectrum.cc
        )
    target_include_directories(splitSpectrum PUBLIC include)
    target_link_libraries(splitSpectrum PUBLIC GDAL::GDAL FFTW::Float OpenMP::OpenMP_CXX)

    InstallSameDir(splitSpectrum)
endif()
