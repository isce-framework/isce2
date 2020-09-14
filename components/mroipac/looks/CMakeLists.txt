Python_add_library(looks MODULE
    bindings/looksmodule.cpp
    )
target_include_directories(looks PUBLIC include)
target_link_libraries(looks PRIVATE
    isce2::DataAccessorLib
    )

InstallSameDir(
    looks
    __init__.py
    Cpxlooks.py
    Looks.py
    Nbymdem.py
    Nbymhgt.py
    Powlooks.py
    Rilooks.py
    )
