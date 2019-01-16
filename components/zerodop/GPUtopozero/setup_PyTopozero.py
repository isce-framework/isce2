#
# Author: Joshua Cohen
# Copyright 2016
#
# This setup file will compile the relevant C++ files against the provided gpu_topozero.pyx
# file to create a gpu_topozero module that can be imported in Python and used as an
# interface for the PyTopozero() object to run the Topo C++ code.

from distutils.core import setup
from distutils.extension import Extension   # Normally not needed but we need to add the
                                            # extra c++11, fopenmp, and lgomp flags
from Cython.Build import cythonize

# Where the .cpp files are located
source_dir = "src/"
# All files contained in source_dir
source_files = ["AkimaLib.cpp",
                "Ellipsoid.cpp",
                "LinAlg.cpp",
                "Orbit.cpp",
                "Peg.cpp",
                "PegTrans.cpp",
                "Poly2d.cpp",
                #"Position.cpp",    Leaving this out for now as it's not being used
                "Topo.cpp",
                "TopoMethods.cpp",
                "Topozero.cpp",
                "UniformInterp.cpp"]
source_files = [(source_dir + f) for f in source_files] # Quick one-line to prepend the source_dir

setup(ext_modules = cythonize(Extension(
        "gpu_topozero",     # Name of the module
        sources=['gpu_topozero.pyx'] + source_files,    # Source files (.cpp and .pyx)
        include_dirs=['include/',   # Header files (.h)
                      '../../iscesys/ImageApi/InterleavedAccessor/include/',
                      '../../iscesys/ImageApi/DataCaster/include/'],
        extra_compile_args=['-fopenmp','-std=c++11','-fPIC','-pthread'],   # Allows for OMP and special libraries
        extra_objects=['gpu-topo.o'],
        extra_link_args=['-lgomp','-lpthread','-L/usr/local/cuda/lib64','-lcudart'],     # Needed to link the OMP/CUDA libraries in
        language="c++"
     )))

