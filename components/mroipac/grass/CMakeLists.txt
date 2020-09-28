isce2_add_cdll(libgrass
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
