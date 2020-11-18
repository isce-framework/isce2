isce2_add_cdll(orbitHermite
    src/orbitHermiteC.c
    src/orbithermite.F
    )
target_include_directories(orbitHermite PUBLIC include)

InstallSameDir(
    orbitHermite
    __init__.py
    Inertial.py
    ODR.py
    Orbit.py
    OrbitExtender.py
    PDS.py
    PRC.py
    Spice.py
    )
