if(TARGET Motif::Motif
   AND TARGET X11::X11
   AND TARGET X11::Xau
   AND TARGET X11::Xt
   )
    add_executable(mdx
        src/graphx_mdx.c
        src/rdf_reader_subs.f
        src/mdx_main.F
        src/mdx.F
        )
    target_compile_definitions(mdx PRIVATE SUN IO64)
    if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
        target_compile_definitions(mdx PRIVATE GFORTRAN)
    endif()
    target_link_libraries(mdx PUBLIC
        Motif::Motif
        X11::X11
        X11::Xau
        X11::Xt
        )
    install(TARGETS mdx
            RUNTIME DESTINATION bin
            )
endif()
