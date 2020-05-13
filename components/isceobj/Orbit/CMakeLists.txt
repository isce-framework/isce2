Python_add_library(orbitHermite MODULE
    src/orbitHermiteC.c
    src/orbithermite.F
    )
target_include_directories(orbitHermite PUBLIC include)

InstallSameDir(
    orbitHermite
    __init__.py
    ODR.py
    Orbit.py
    OrbitExtender.py
    PDS.py
    PRC.py
    Spice.py
    )
