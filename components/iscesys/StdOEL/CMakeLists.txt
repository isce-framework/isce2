add_subdirectory(src)
target_include_directories(stdoelLib PUBLIC include)

Python_add_library(StdOEL MODULE
    bindings/StdOELmodule.cpp
    )
target_link_libraries(StdOEL PUBLIC isce2::stdoelLib)

InstallSameDir(
    StdOEL
    __init__.py
    StdOELPy.py
    )

add_executable(testStdOEL test/testStdOEL.cpp)
target_include_directories(testStdOEL PUBLIC include)
target_link_libraries(testStdOEL PRIVATE isce2::stdoelLib)
add_exe_test(testStdOEL)
