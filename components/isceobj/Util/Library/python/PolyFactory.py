#!/usr/bin/env python3
import isce
import sys
from isceobj.Util.Poly2D import Poly2D
from isceobj.Util.Poly1D import Poly1D

def createPoly(polyType = '2d',family=None,name=None):
    pol = None
    if polyType == '2d':
        pol = Poly2D(family,name)
    else:
        pol = Poly1D(family,name)
    return pol
if __name__ == '__main__':
    sys.exit(main())
