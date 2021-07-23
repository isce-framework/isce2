if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/RelaxIV)
    Python_add_library(unwcomp MODULE
        bindings/unwcompmodule.cpp
        src/RelaxIV/RelaxIV.C
        src/relaxIVdriver.cpp
        )
    target_include_directories(unwcomp PUBLIC include)
    InstallSameDir(
    __init__.py
    phaseUnwrap.py
    unwrapComponents.py
    unwcomp
    )
else()
    InstallSameDir(
    __init__.py
    phaseUnwrap.py
    unwrapComponents.py
    )
endif()
