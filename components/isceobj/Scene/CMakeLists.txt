add_library(concatenate SHARED
    src/frame_concatenate.c
    src/swst_resample.c
    )

InstallSameDir(
    concatenate
    __init__.py
    Frame.py
    Track.py
    )
