add_subdirectory(data)
add_subdirectory(language)
add_subdirectory(reserved)
add_subdirectory(units)

InstallSameDir(
    __init__.py
    eRDF.py
    iRDF.py
    parse.py
    read.py
    uRDF.py
    utils.py
    )
