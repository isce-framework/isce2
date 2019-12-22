Python_add_library(correlationlib MODULE
    bindings/correlationmodule.cpp
    src/cchz_wave.cpp
    src/magnitude_threshold.c
    )
target_include_directories(correlationlib PUBLIC include)
target_link_libraries(correlationlib PRIVATE
    DataAccessor_static
    )

InstallSameDir(
    correlationlib
    __init__.py
    correlation.py
    )
