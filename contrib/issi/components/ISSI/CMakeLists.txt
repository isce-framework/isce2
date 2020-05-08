add_library(issi SHARED
    src/calculateBVector.c
    src/cfr.c
    src/cfrToFr.c
    src/frToTEC.c
    src/igrf2005_sub.f
    src/polarimetricCalibration.f
    src/polcal.c
    src/tecToPhase.c
    )
target_include_directories(issi PUBLIC include)

add_subdirectory(src)

InstallSameDir(
    issi
    __init__.py
    FR.py
    )
