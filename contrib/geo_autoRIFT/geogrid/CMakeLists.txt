Python_add_library(geogrid MODULE
    bindings/geogridmodule.cpp
    src/geogrid.cpp
    )
target_include_directories(geogrid PRIVATE
    include
    )
target_link_libraries(geogrid PRIVATE
    GDAL::GDAL
    combinedLib
    )

InstallSameDir(
    geogrid
    __init__.py
    Geogrid.py
    GeogridOptical.py
    )
