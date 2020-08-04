Python_add_library(ALOS_fbs2fbd MODULE
    bindings/ALOS_fbs2fbdmodule.c
    )
target_include_directories(ALOS_fbs2fbd PUBLIC include)
target_link_libraries(ALOS_fbs2fbd PUBLIC
    alos_lib
    utilLib
    )

InstallSameDir(
    ALOS_fbs2fbd
    __init__.py
    ALOS_fbs2fbdPy.py
    )
