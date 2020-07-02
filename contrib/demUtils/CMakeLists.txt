Python_add_library(correct_geoid_i2_srtm MODULE
    correct_geoid_i2_srtm/bindings/correct_geoid_i2_srtmmodule.cpp
    correct_geoid_i2_srtm/src/correct_geoid_i2_srtm.f
    correct_geoid_i2_srtm/src/correct_geoid_i2_srtmState.f
    correct_geoid_i2_srtm/src/correct_geoid_i2_srtmSetState.f
    )
target_include_directories(correct_geoid_i2_srtm PUBLIC
    correct_geoid_i2_srtm/include
    )
target_link_libraries(correct_geoid_i2_srtm PUBLIC
    DataAccessor_static
    stdoel_static
    utilLib
    )

Python_add_library(demStitch MODULE
    demstitcher/bindings/demStitch.c
    )

Python_add_library(upsampledem MODULE
    upsampledem/bindings/upsampledemmodule.cpp
    upsampledem/src/upsampledem.f
    upsampledem/src/upsampledemSetState.f
    upsampledem/src/upsampledemState.f
    )
target_include_directories(upsampledem PRIVATE
    upsampledem/include
    )
target_link_libraries(upsampledem PRIVATE
    utilLib
    )

Python_add_library(watermask MODULE
    watermask/bindings/watermaskmodule.cpp
    watermask/src/watermask.cpp
    )
target_include_directories(watermask PRIVATE
    watermask/include
    )

InstallSameDir(
    demStitch
    upsampledem
    watermask
    correct_geoid_i2_srtm
    __init__.py
    correct_geoid_i2_srtm/Correct_geoid_i2_srtm.py
    correct_geoid_i2_srtm/egm96geoid.dat
    demstitcher/DemStitcher.py
    demstitcher/DemStitcherV3.py
    swbdstitcher/SWBDStitcher.py
    upsampledem/UpsampleDem.py
    watermask/WaterMask.py
    )
