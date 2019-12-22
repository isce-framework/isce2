find_package(X11)

if(X11_FOUND)

    foreach(component
            Xmu
            Xt
            )

        if(X11_${Xmu}_FOUND AND NOT TARGET X11::${component})
            add_library(X11::${component} IMPORTED INTERFACE)
            target_include_directories(X11::${component} SYSTEM
                INTERFACE ${X11_${component}_INCLUDE_PATH})
            target_link_libraries(X11::${component}
                INTERFACE ${X11_${component}_LIB})
        endif()
    endforeach()
endif()
