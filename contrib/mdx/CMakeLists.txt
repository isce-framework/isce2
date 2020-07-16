if(TARGET Motif::Motif
   AND TARGET X11::X11
   AND TARGET X11::Xau
   AND TARGET X11::Xt
   )

    #Only for this folder - ensure openmp is not used for compiling
    #Modifying flags here have directory scope
    if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
        string(REPLACE "-fopenmp" " " CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS}")
    endif()
    
    add_executable(mdx
        src/graphx_mdx.c
        src/rdf_reader_subs.f
        src/mdx_main.F
        src/mdx_subs.F
        )
    target_compile_definitions(mdx PRIVATE SUN IO64)
    target_include_directories(mdx PUBLIC 
        ${X11_Xt_INCLUDE_PATH}
        ${X11_X11_INCLUDE_PATH}
        ${MOTIF_INCLUDE_DIR})
    target_link_libraries(mdx PUBLIC
        ${X11_X11_LIB}
        ${MOTIF_LIBRARIES}
        ${X11_Xt_LIB}
        )

    install(TARGETS mdx
            RUNTIME DESTINATION bin
            )
endif()
