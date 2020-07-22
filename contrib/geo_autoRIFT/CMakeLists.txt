# Early exit if prereqs not found
if(NOT TARGET Python::NumPy
OR NOT TARGET GDAL::GDAL
OR NOT OpenCV_FOUND
   )
    return()
endif()

InstallSameDir(__init__.py)

add_subdirectory(autoRIFT)
add_subdirectory(geogrid)
