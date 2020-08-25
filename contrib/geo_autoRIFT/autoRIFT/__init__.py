#!/usr/bin/env python

#def SplitRangeSpectrum():
#    from .splitSpectrum import PySplitRangeSpectrum
#    return PySplitRangeSpectrum()

# should always work - standalone or with ISCE
from .autoRIFT import autoRIFT

try:
    from .autoRIFT_ISCE import autoRIFT_ISCE
except ImportError:
    # this means ISCE support not available. Don't raise error. Allow standalone use
    pass
