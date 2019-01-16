#!/usr/bin/env python

def Ellipsoid(a=None, e2=None):
    from .isceLib import PyEllipsoid
    return PyEllipsoid(a,e2)

def Peg(lat=None, lon=None, hdg=None):
    from .isceLib import PyPeg
    return PyPeg(lat,lon,hdg)

def Pegtrans():
    from .isceLib import PyPegtrans
    return PyPegtrans()

def Position():
    from .isceLib import PyPosition
    return PyPosition()

def LinAlg():
    from .isceLib import PyLinAlg
    return PyLinAlg()

def Poly1d(order=None, mean=0., norm=1., coeffs=None):
    from .isceLib import PyPoly1d
    return PyPoly1d(order,mean,norm,coeffs)

def Poly2d(azimuthOrder=None, rangeOrder=None, azimuthMean=0., rangeMean=0., azimuthNorm=1., rangeNorm=1., coeffs=None):
    from .isceLib import PyPoly2d
    return PyPoly2d(azimuthOrder,rangeOrder,azimuthMean,rangeMean,azimuthNorm,rangeNorm,coeffs)

def Orbit(basis=None, nVectors=None):
    from .isceLib import PyOrbit
    return PyOrbit(basis,nVectors)
