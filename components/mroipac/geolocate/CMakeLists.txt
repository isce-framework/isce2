isce2_add_cdll(libgeolocate
    src/geolocate_wrapper.c
    src/geolocate.f
    )

InstallSameDir(
    libgeolocate
    __init__.py
    Geolocate.py
    )
