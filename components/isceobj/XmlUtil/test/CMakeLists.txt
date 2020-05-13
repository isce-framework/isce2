# TODO add_python_test(testXmlUtilPy.py)

foreach(xml
        test1.xml
        )

    configure_file(${xml} ${xml})
endforeach()
