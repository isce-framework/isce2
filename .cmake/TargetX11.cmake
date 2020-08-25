set(components
    Xau
    Xt
    )

find_package(X11 COMPONENTS ${components})

if(X11_FOUND)

    # make X11 look like a regular find_package component
    set(X11_X11_FOUND TRUE)
    set(X11_X11_INCLUDE_PATH ${X11_INCLUDE_DIR})
    list(APPEND components X11)

    foreach(component ${components})
        if(X11_${component}_FOUND AND
           NOT TARGET X11::${component})
            add_library(X11::${component} IMPORTED INTERFACE)
            target_link_libraries(X11::${component}
                INTERFACE ${X11_${component}_LIB})
            target_include_directories(X11::${component} SYSTEM
                INTERFACE ${X11_${component}_INCLUDE_PATH})
        endif()
    endforeach()
endif()
