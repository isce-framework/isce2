Python_add_library(snaphu MODULE
    bindings/snaphumodule.cpp
    src/snaphu.c
    src/snaphu_cost.c
    src/snaphu_cs2.c
    src/snaphu_io.c
    src/snaphu_solver.c
    src/snaphu_tile.c
    src/snaphu_util.c
    )
target_include_directories(snaphu PUBLIC include)
target_link_libraries(snaphu PUBLIC
    DataAccessor_static
    )

InstallSameDir(
    snaphu
    __init__.py
    Snaphu.py
    )
