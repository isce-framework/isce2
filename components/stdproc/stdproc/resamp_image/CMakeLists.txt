set(CMAKE_POSITION_INDEPENDENT_CODE ON)

Python_add_library(resamp_image MODULE
    bindings/resamp_imagemodule.cpp
    src/resamp_imageState.F
    src/resamp_image.f90
    src/resamp_imageAllocateDeallocate.F
    src/resamp_imageSetState.F
    )
target_include_directories(resamp_image PUBLIC include)
target_link_libraries(resamp_image PUBLIC
    isce2::DataAccessorLib
    isce2::resampLib
    isce2::utilLib
    isce2::stdoelLib
    )

InstallSameDir(
    resamp_image
    __init__.py
    Resamp_image.py
    )
