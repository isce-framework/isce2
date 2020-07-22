Python_add_library(autoriftcore MODULE
    bindings/autoriftcoremodule.cpp
    )
target_include_directories(autoriftcore PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
    )
target_link_libraries(autoriftcore PRIVATE
    Python::NumPy
    ${OpenCV_LIBS}
    )

InstallSameDir(
    autoriftcore
    __init__.py
    autoRIFT_ISCE.py
    autoRIFT.py
    )
