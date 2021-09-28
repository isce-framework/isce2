add_subdirectory(issi/components/ISSI ISSI)
add_subdirectory(Snaphu)
add_subdirectory(demUtils)
add_subdirectory(frameUtils)
add_subdirectory(downsample_unwrapper)
add_subdirectory(geo_autoRIFT)
add_subdirectory(PyCuAmpcor)
add_subdirectory(splitSpectrum)
add_subdirectory(alos2filter)
add_subdirectory(alos2proc)
add_subdirectory(alos2proc_f)
add_subdirectory(rfi)
add_subdirectory(mdx)
add_subdirectory(UnwrapComp)
InstallSameDir(
    __init__.py
    )
install(PROGRAMS issi/applications/ISSI.py
    DESTINATION ${ISCE2_PKG}/applications
    )
