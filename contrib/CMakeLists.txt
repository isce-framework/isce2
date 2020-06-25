add_subdirectory(issi)
add_subdirectory(Snaphu)
add_subdirectory(demUtils)
add_subdirectory(frameUtils)
#add_subdirectory(unwUtils)
add_subdirectory(downsample_unwrapper)

add_subdirectory(PyCuAmpcor)
add_subdirectory(splitSpectrum)
add_subdirectory(alos2filter)
add_subdirectory(alos2proc)
add_subdirectory(alos2proc_f)
add_subdirectory(rfi)
add_subdirectory(mdx)

InstallSameDir(
    __init__.py
    )
