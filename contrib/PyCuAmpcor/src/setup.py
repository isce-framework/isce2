#
# Implementation: python setup_cuAmpcor.py build_ext --inplace
# Generates PyCuAmpcor.xxx.so (where xxx is just some local sys-arch information).
# Note you need to run your makefile *FIRST* to generate the cuAmpcor.o object.
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

os.environ["CC"] = "g++"

setup(  name = 'PyCuAmpcor',
        ext_modules = cythonize(Extension(
        "PyCuAmpcor",
        sources=['PyCuAmpcor.pyx'],
        include_dirs=['/usr/local/cuda/include'], # REPLACE WITH YOUR PATH TO YOUR CUDA LIBRARY HEADERS
        extra_compile_args=['-fPIC','-fpermissive'],
        extra_objects=['SlcImage.o','cuAmpcorChunk.o','cuAmpcorParameter.o','cuCorrFrequency.o',
                       'cuCorrNormalization.o','cuCorrTimeDomain.o','cuArraysCopy.o',
                       'cuArrays.o','cuArraysPadding.o','cuOffset.o','cuOverSampler.o',
                       'cuSincOverSampler.o', 'cuDeramp.o','cuAmpcorController.o'],
        extra_link_args=['-L/usr/local/cuda/lib64','-lcuda','-lcudart','-lcufft','-lcublas'], # REPLACE FIRST PATH WITH YOUR PATH TO YOUR CUDA LIBRARIES
        language='c++'
    )))
