add_library(stdoel_static STATIC
    src/ScreenWriter.cpp
    src/StdOELF.cpp
    src/FileWriter.cpp
    src/StdOEL.cpp
    src/WriterFactory.cpp
    )
set_property(TARGET stdoel_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(stdoel_static PUBLIC include)

Python_add_library(StdOEL MODULE
    bindings/StdOELmodule.cpp
    )
target_link_libraries(StdOEL PUBLIC stdoel_static)

InstallSameDir(
    StdOEL
    __init__.py
    StdOELPy.py
    )

add_executable(testStdOEL test/testStdOEL.cpp
    src/ScreenWriter.cpp
    src/StdOELF.cpp
    src/FileWriter.cpp
    src/StdOEL.cpp
    src/WriterFactory.cpp
    )
target_include_directories(testStdOEL PUBLIC include)
add_exe_test(testStdOEL)
