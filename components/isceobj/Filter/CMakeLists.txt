Python_add_library(filter MODULE
    bindings/filtermodule.cpp
    src/cpxPhaseFilter.cpp
    src/EdgeFilter.cpp
    src/Filter.cpp
    src/filterPhase.cpp
    src/GaussianFilter.cpp
    src/MeanFilter.cpp
    src/medianFilter.cpp
    )
target_include_directories(filter PUBLIC include)
InstallSameDir(
    __init__.py
    Filter.py
    filter
    )
