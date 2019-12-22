find_package(GDAL)

# Make a compatibility GDAL::GDAL interface target
# In CMake >= 3.14, this already exists for us :)
if(GDAL_FOUND AND NOT TARGET GDAL::GDAL)
    add_library(GDAL::GDAL IMPORTED INTERFACE)
    target_include_directories(GDAL::GDAL SYSTEM INTERFACE ${GDAL_INCLUDE_DIRS})
    target_link_libraries(GDAL::GDAL INTERFACE ${GDAL_LIBRARIES})
endif()
