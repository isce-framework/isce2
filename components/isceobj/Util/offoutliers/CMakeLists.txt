add_subdirectory(src)
target_include_directories(offoutliersLib PUBLIC include)
target_link_libraries(offoutliersLib PRIVATE isce2::stdoelLib)
