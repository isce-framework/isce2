#!/usr/bin/env python

#def SplitRangeSpectrum():
#    from .splitSpectrum import PySplitRangeSpectrum
#    return PySplitRangeSpectrum()

# should always work - standalone or with ISCE
from .GeogridOptical import GeogridOptical

try:
    from .Geogrid import Geogrid
except ImportError:
    # this means ISCE support not available. Don't raise error. Allow standalone use
    pass
