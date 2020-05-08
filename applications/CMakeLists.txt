set(files
    __init__.py
    alos2App.py
    alos2burstApp.py
    dataTileManager.py
    dem.py
    demdb.py
    downsampleDEM.py
    fixImageXml.py
    gdal2isce_xml.py
    imageMath.py
    insarApp.py
    isce2geotiff.py
    isce2gis.py
    isceApp.py
    iscehelp.py
    looks.py
    make_raw.py
    mdx.py
    rtcApp.py
    stitcher.py
    stripmapApp.py
    topsApp.py
    upsampleDem.py
    waterMask.py
    wbd.py
    wbdStitcher.py
    )

install(PROGRAMS ${files}
        DESTINATION ${ISCE2_PKG}/applications)

# Symlink apps into PREFIX/bin so they are on the $PATH
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory \
    ${CMAKE_INSTALL_PREFIX}/bin)"
    )
foreach(file ${files})
    install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/${ISCE2_PKG}/applications/${file} \
        ${CMAKE_INSTALL_PREFIX}/bin/${file})"
        )
endforeach()
