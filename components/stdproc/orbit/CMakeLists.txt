add_subdirectory(orbitLib)

add_subdirectory(fdmocomp)
add_subdirectory(getpeg)
add_subdirectory(mocompbaseline)
add_subdirectory(orbit2sch)
add_subdirectory(sch2orbit)
add_subdirectory(setmocomppath)

InstallSameDir(
    __init__.py
    pegManipulator.py
    )
