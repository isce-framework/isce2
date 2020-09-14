Python_add_library(formslc MODULE
    formslc/bindings/formslcmodule.cpp
    formslc/src/formslc.F
    formslc/src/formslcGetState.F
    formslc/src/formslcSetState.F
    formslc/src/formslcState.F
    src/acpatch.F
    src/intp_coef.F
    src/rcpatch.F
    src/rmpatch.F
    )
target_include_directories(formslc PUBLIC
    formslc/include
    )
target_link_libraries(formslc PRIVATE
    isce2::DataAccessorLib
    )

target_link_libraries(formslc PUBLIC
    isce2::utilLib
    )

InstallSameDir(
    formslc
    formslc/__init__.py
    formslc/FormSLC.py
    )

set(tests
    formslc/test/driverFormslc.py
    formslc/test/testFormslcPy.py
    formslc/test/Platform930110.xml
    formslc/test/SlcImage930110.xml
    formslc/test/platform950523Init.ini
    formslc/test/DriverFormSLC.xml
    formslc/test/DriverFormSLCXXX.xml
    formslc/test/Radar930110.xml
    formslc/test/RawImage930110.xml
    formslc/test/SlcImage930110New.xml
    formslc/test/exampleCommandLine
    formslc/test/formslcInit.ini
    formslc/test/platform930110Init.ini
    )
