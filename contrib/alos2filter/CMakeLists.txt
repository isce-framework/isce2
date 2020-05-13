add_library(alos2filter SHARED src/psfilt1.c)
set_target_properties(alos2filter PROPERTIES
    PREFIX ""
    OUTPUT_NAME libalos2filter
    SUFFIX .so)
target_link_libraries(alos2filter PUBLIC FFTW::Float)

InstallSameDir(
    alos2filter
    __init__.py
    alos2filter.py
    )
