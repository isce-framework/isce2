Python_add_library(geogrid MODULE
    bindings/geogridmodule.cpp
    src/geogrid.cpp
    )
target_include_directories(geogrid PRIVATE
    include
    )
target_link_libraries(geogrid PRIVATE
    GDAL::GDAL
    isce2::combinedLib
    )
    
Python_add_library(geogridOptical MODULE
    bindings/geogridOpticalmodule.cpp
    src/geogridOptical.cpp
    )
target_include_directories(geogridOptical PRIVATE
    include
    )
target_link_libraries(geogridOptical PRIVATE
    GDAL::GDAL
    isce2::combinedLib
    )

InstallSameDir(
    geogridOptical
    geogrid
    __init__.py
    Geogrid.py
    GeogridOptical.py
    )
