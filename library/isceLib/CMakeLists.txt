cython_add_module(isceLib
    pyx/isceLib.pyx
    src/Ellipsoid.cpp
    src/LinAlg.cpp
    src/Orbit.cpp
    src/Peg.cpp
    src/Pegtrans.cpp
    src/Poly1d.cpp
    src/Poly2d.cpp
    src/Position.cpp
    )
target_include_directories(isceLib PUBLIC include)

InstallSameDir(
    isceLib
    __init__.py
    )
