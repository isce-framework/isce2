InstallSameDir(
    __init__.py
    StdOEPy.py
    )

isce2_add_staticlib(stdoeLib
    src/StdOE.cpp
    src/StdOEDefaults.cpp
    src/StdOEF.cpp
    )
target_include_directories(stdoeLib PUBLIC
    include
    )

Python_add_library(StdOE MODULE
    bindings/StdOEmodule.cpp
    )
target_link_libraries(StdOE PRIVATE isce2::stdoeLib)
