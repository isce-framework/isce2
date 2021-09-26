add_library(issi SHARED
    src/calculateBVector.c
    src/cfr.c
    src/cfrToFr.c
    src/frToTEC.c
    src/igrf2005_sub.f
    src/polarimetricCalibration.f
    src/polcal.c
    src/tecToPhase.c
    )
set_target_properties(issi PROPERTIES
    PREFIX ""
    OUTPUT_NAME issi
    SUFFIX .so)
target_include_directories(issi PUBLIC include)

#file(RELATIVE_PATH relpath ${isce2_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
install(DIRECTORY src/igrf_data
        DESTINATION ${ISCE2_PKG}/components/contrib/ISSI
        )
install(PROGRAMS FR.py
        DESTINATION ${ISCE2_PKG}/components/contrib/ISSI
        )
install(PROGRAMS __init__.py
        DESTINATION ${ISCE2_PKG}/components/contrib/ISSI
        )
install(TARGETS issi
        DESTINATION ${ISCE2_PKG}/components/contrib/ISSI
        )
