add_library(LineAccessor_static STATIC
    src/ImageAccessor.cpp
    src/LineAccessor.cpp
    src/LineAccessorF.cpp
    )
target_include_directories(LineAccessor_static PUBLIC include)
set_property(TARGET LineAccessor_static PROPERTY POSITION_INDEPENDENT_CODE ON)

Python_add_library(LineAccessor MODULE
    bindings/LineAccessormodule.cpp
    )
target_link_libraries(LineAccessor PUBLIC LineAccessor_static)

InstallSameDir(
    LineAccessor
    __init__.py
    LineAccessorPy.py
    )
