Python_add_library(icu MODULE
    bindings/icumodule.cpp
    src/grass.F
    src/icu.F
    src/icuState.F
    src/residues.F
    src/rt.F
    src/unw_rt.F
    src/SConscript
    src/abs_phase.F
    src/bermuda.F
    src/gen_neutrons.F
    src/icuSetState.F
    src/intf_cc.F
    src/intf_filt.F
    src/lowpass.F
    src/norm_cor.F
    src/ph_sigma.F
    src/ph_slope.F
    src/psfilt_sub.F
    src/std_cor.F
    )
target_include_directories(icu PUBLIC include)
target_link_libraries(icu PUBLIC
    isce2::DataAccessorLib
    isce2::utilLib
    )

InstallSameDir(
    icu
    __init__.py
    Icu.py
    )
