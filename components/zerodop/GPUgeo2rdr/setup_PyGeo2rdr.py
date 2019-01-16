#
# Author: Joshua Cohen
# Copyright 2017
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

source_dir = "src/"
source_files = ["Ellipsoid.cpp",
                "Geo2rdr.cpp",
                "GeoController.cpp",
                "LinAlg.cpp",
                "Orbit.cpp",
                "Poly1d.cpp"]
source_files = [(source_dir + f) for f in source_files]

setup(ext_modules = cythonize(Extension(
        "GPUgeo2rdr",
        sources=['GPUgeo2rdr.pyx'] + source_files,
        include_dirs=['include/',
                      '/home/joshuac/isce/build/GPUisce/components/iscesys/ImageApi/include',
                      '/home/joshuac/isce/build/iscesys/ImageApi/DataCaster/include/'],
        extra_compile_args=['-fopenmp','-O3','-std=c++11','-fPIC','-pthread'],
        extra_objects=['GPUgeo.o'],
        extra_link_args=['-lgomp','-L/usr/local/cuda/lib64','-lcudart','-L/home/joshuac/isce/build/gpu-isce/libs/','-lDataAccessor','-lInterleavedAccessor','-lcombinedLib','-lgdal'],
        language="c++"
     )))
