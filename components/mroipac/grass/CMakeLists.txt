Python_add_library(libgrass MODULE
    src/corr_flag.c
    src/grass.c
    src/trees.c
    src/residue.c
    )

InstallSameDir(
    libgrass
    __init__.py
    grass.py
    )
