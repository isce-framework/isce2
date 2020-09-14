isce2_add_staticlib(LineAccessorLib
    src/ImageAccessor.cpp
    src/LineAccessor.cpp
    src/LineAccessorF.cpp
    )
target_include_directories(LineAccessorLib PUBLIC include)

Python_add_library(LineAccessor MODULE
    bindings/LineAccessormodule.cpp
    )
target_link_libraries(LineAccessor PRIVATE isce2::LineAccessorLib)

InstallSameDir(
    LineAccessor
    __init__.py
    LineAccessorPy.py
    )
