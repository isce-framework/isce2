InstallSameDir(
    __init__.py
    CalcSchHeightVel.py
    )

if(NOT ISCE2_WITH_STANFORD)
    return()
endif()

isce2_add_staticlib(orbitLib
    src/ave_tpsch.f90
    src/convert_sch_to_xyz.F
    src/convert_schdot_to_xyzdot.F
    src/curvature.F
    src/dot.f90
    src/geo_hdg.F
    src/get_tpsch.f90
    src/latlon.F
    src/lincomb.F
    src/matmat.F
    src/matvec.F
    src/radar_to_xyz.F
    src/schbasis.F
    src/tranmat.F
    src/unitvec.f90
    )
