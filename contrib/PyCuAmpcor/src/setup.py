#
# Implementation: python setup_cuAmpcor.py build_ext --inplace
# Generates PyCuAmpcor.xxx.so (where xxx is just some local sys-arch information).
# Note you need to run your makefile *FIRST* to generate the cuAmpcor.o object.
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

setup(  name = 'PyCuAmpcor',
        ext_modules = cythonize(Extension(
        "PyCuAmpcor",
        sources=['PyCuAmpcor.pyx'],
        include_dirs=['/usr/local/cuda/include', numpy.get_include()], # REPLACE WITH YOUR PATH TO YOUR CUDA LIBRARY HEADERS
        extra_compile_args=['-fPIC','-fpermissive'],
        extra_objects=['GDALImage.o','cuAmpcorChunk.o','cuAmpcorParameter.o','cuCorrFrequency.o',
                       'cuCorrNormalization.o','cuCorrTimeDomain.o','cuArraysCopy.o',
                       'cuArrays.o','cuArraysPadding.o','cuOffset.o','cuOverSampler.o',
                       'cuSincOverSampler.o', 'cuDeramp.o','cuAmpcorController.o','cuEstimateStats.o'],
        extra_link_args=['-L/usr/local/cuda/lib64',
                        '-L/usr/lib64/nvidia',
                        '-lcuda','-lcudart','-lcufft','-lcublas','-lgdal'], # REPLACE FIRST PATH WITH YOUR PATH TO YOUR CUDA LIBRARIES
        language='c++'
    )))
