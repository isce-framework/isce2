import os
import copy
import ctypes
import logging
import isceobj
from xml.etree.ElementTree import ElementTree

def psfilt1(inputfile, outputfile, width, alpha, fftw, step):
    filters = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libalos2filter.so'))
    filters.psfilt1(
        ctypes.c_char_p(bytes(inputfile,'utf-8')),
        ctypes.c_char_p(bytes(outputfile,'utf-8')),
        ctypes.c_int(width),
        ctypes.c_double(alpha),
        ctypes.c_int(fftw),
        ctypes.c_int(step)
        )
