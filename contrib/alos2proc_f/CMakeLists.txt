if(NOT CYTHON_EXECUTABLE)
    return()
endif()

cython_add_module(alos2proc_f
    pyx/alos2proc_f.pyx
    src/bilinear.f
    src/curvature.f
    src/interp.f
    src/lincomb.f
    src/matvec.f
    src/rect.f
    src/convert_sch_to_xyz.f
    src/enubasis.f
    src/intpcoefnorm.f
    src/look_coord_conv.f
    src/norm.f
    src/rect_with_looks.f
    src/tranmat.f
    src/cross.f
    src/fitoff.f
    src/latlon.f
    src/matmat.f
    src/radar_to_xyz.f
    src/schbasis.f
    src/cbind.f90
    )

target_include_directories(alos2proc_f PUBLIC include)
target_link_libraries(alos2proc_f PUBLIC FFTW::Float OpenMP::OpenMP_Fortran)

InstallSameDir(
    alos2proc_f
    __init__.py
    )
