Python_add_library(libfilter MODULE
    src/rescale_magnitude.c
    src/psfilt.c
    src/timing.c
    )
target_include_directories(libfilter PUBLIC include)

InstallSameDir(
    libfilter
    __init__.py
    Filter.py
    )
