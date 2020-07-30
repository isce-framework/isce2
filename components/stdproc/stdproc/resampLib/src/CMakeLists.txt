isce2_add_staticlib(resampLib
    poly_funcs.f90
    svd.f
    svdvecfit.f
    )
target_link_libraries(resampLib PRIVATE
    isce2::utilLib
    )
