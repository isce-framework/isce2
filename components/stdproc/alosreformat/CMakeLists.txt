add_library(alos_lib STATIC
    ALOS_lib/src/cfft1d.c
    ALOS_lib/src/find_fft_length.c
    ALOS_lib/src/utils.c
    ALOS_lib/src/ALOS_ldr_orbit.c
    ALOS_lib/src/SConscript
    ALOS_lib/src/calc_dop.c
    ALOS_lib/src/cfft1d_fftpack.c
    ALOS_lib/src/fftpack.c
    ALOS_lib/src/get_sio_struct.c
    ALOS_lib/src/hermite_c.c
    ALOS_lib/src/interpolate_ALOS_orbit.c
    ALOS_lib/src/null_sio_struct.c
    ALOS_lib/src/read_ALOS_sarleader.c
    ALOS_lib/src/rng_compress.c
    )
target_include_directories(alos_lib PUBLIC include)
set_property(TARGET alos_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(ALOS_fbd2fbs)
add_subdirectory(ALOS_fbs2fbd)

InstallSameDir(__init__.py)
