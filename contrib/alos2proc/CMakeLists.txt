Python_add_library(libalos2proc MODULE
    src/rg_filter.c
    src/lib_file.c
    src/lib_cpx.c
    src/mbf.c
    src/lib_array.c
    src/lib_func.c
    )
target_include_directories(libalos2proc PUBLIC include)
target_link_libraries(libalos2proc PUBLIC
    FFTW::Float
    )

InstallSameDir(
    libalos2proc
    __init__.py
    alos2proc.py
    )
