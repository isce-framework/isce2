Python_add_library(ALOS_fbd2fbs MODULE
    bindings/ALOS_fbd2fbsmodule.c
    )
target_include_directories(ALOS_fbd2fbs PUBLIC include)
target_link_libraries(ALOS_fbd2fbs PUBLIC
    alos_lib
    utilLib
    )

InstallSameDir(
    ALOS_fbd2fbs
    __init__.py
    ALOS_fbd2fbsPy.py
    )
