set(files
    __init__.py
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
    scansarApp.py
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
