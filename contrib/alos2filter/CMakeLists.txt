add_library(alos2filter SHARED src/psfilt1.c)
InstallSameDir(
    alos2filter
    __init__.py
    alos2filter.py
    )
