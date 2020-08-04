Python_add_library(libgeolocate MODULE
    src/geolocate_wrapper.c
    src/geolocate.f
    )

InstallSameDir(
    libgeolocate
    __init__.py
    Geolocate.py
    )
