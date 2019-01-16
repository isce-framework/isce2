#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# United States Government Sponsorship acknowledged. This software is subject to
# U.S. export control laws and regulations and has been classified as 'EAR99 NLR'
# (No [Export] License Required except when exporting to an embargoed country,
# end user, or in support of a prohibited end use). By downloading this software,
# the user agrees to comply with all applicable U.S. export laws and regulations.
# The user has the responsibility to obtain export licenses, or other export
# authority as may be required before exporting this software to any 'EAR99'
# embargoed foreign country or citizen of those countries.
#
# Author: Eric Belz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



"""Affine
======

Last but not least is the Affine transformation. This is made up of a
rotation  and a translation. The rotation can be any a Matrix, Versor,
EulerAngle, YPR, ... or something else, as long as it's call method takes
the x, y, z attributes where they need to be). They key is in the operator
overalods:

__invert__  --> "~"  --> inverse transformation

x', y', z' = A(x, y, z) goes from ECEF to tangent plane cartesian, then

x, y, z = (~A)(x', y', z') goes from tangent plane to ECEF.

That is really handy. The coordinate transform is a 1st class object, and
it's invertible via an operator.

__mul__  -->  "*"  --> composition of transformations

So, for instance, you can go from ECEF to an airplane with A, the go
from the platforms's IMU and the Antenna with A', then

A" = A*A'

will go from ECEF to your antenna, including the motion. It's really that
simple.


__call__ --> ()   -> applies the transformation to the object.

With the affine transformation's compose() method, you can build any
transformation you want from intermediate steps. Support for a and
aircraft frame is forthcoming-- on in which you diffrentiate a motion
history from a GPS record, define a velocity-tangent frame, and then
correct for platform attitude.

Final Note:
----------
With the helmert() function, you get a standard geodesy affine transformation
that includes a scaling factor-- all this means is that you have to use a
Matrix (Tensor) for the now mis-named "rotation" attribute.

Nothing is type checked-- so you are responsible for your transformations.
"""
## \namespace geo::affine Affine Transformations
from isceobj.Util.geo import euclid

## Limited <a href="http://en.wikipedia.org/wiki/Affine_transformation">Affine
## Transformations</a>.
class Affine(euclid.Alias):
    """Affine(rotation, translation)

    rotation: perferably a euclid.chart on SO3
    translation:           euclid.Vector in E3

    Methods:
    ========

    A(v)    applies transformation
    A*A'    composes transformations [see Note]
    ~A      returns the inverse transformation


    NOTE: A and A' need to have their rotation attribute be the same class for
    composition to work. (You can't multiply a versor and an euler angle)--
    also, if you're doing scaling, skewing, or gliding -- you better use a
    Matrix (Tensor).
    """
    ## Init: a callable rotation and translation
    def __init__(self, rotation, translation):
        """see class docstring for signature"""
        ## Alias rotation (or not, you could make it a shear, dilation, ...)
        self.rotation = rotation
        ## Translation
        self.translation = translation
        return None

    ## Affine transform:\n
    ## \f$ A(\vec{v}) \rightarrow \vec{v}' = R(\vec{v}) + \vec{T} \f$
    def __call__(self, vector):
        """vector = affine(vector)  ==>

        affine.translation + affine.rotation(vector)"""
        return self.translation+self.rotation(vector)

    ## Convolution \n
    ## \f$ AA' = (R, T)(R', T') \rightarrow (RR', R(T') + T) \f$ \n
    def __mul__(self, other):
        """A*A' = (R, T)*(R', T') --? (R*R', R(T') + T)
        is the composition of two affine transformations.
        """
        return self.__class__(
            self.rotation*other.rotation,
            self.rotation*other.translation + self.translation
            )

    ## Inverse \n
    ## \f$ AA^{-1} = ({\bf 1}, 0) \rightarrow A^{-1} = (R^{-1}, -R^{-1}(T)) \f$
    def __invert__(self):
        """~A = ~(R, T) --> (~R, -(~R(T)))

        is the affine transformation such that:

        (~A)*(A) == 1"""
        inv_rot = ~(self.rotation)
        return self.__class__(inv_rot, -(inv_rot(self.translation)))

    pass


## \f$ \vec{v}' = \vec{C} + [\mu {\bf I} + \vec{r} {\bf \times}]\vec{v}  \f$ \n
## A <a href="http://en.wikipedia.org/wiki/Helmert_transformation">Helmert
## </a> transformation.
def helmert(cx, cy, cz, s, rx, ry, rz):
    """
    affine = Helmert(C, s, rx, ry, rz)
    cx, cy, cz  in meters (a Vector)
    mu in ppm
    rx, ry, rz in arcseconds (*r as a Vector-- since it is a small rotation)
    """
    from .euclid import IDEM, Vector
    from math import pi


    C = Vector(*map(float, (cx, cy, cz)))
    # note: sign is correct, since R will take an anterior product,
    # recall dual() makes a matrix representing the cross product.
    R = (Vector(rx, ry, rz)*pi/180./3600.).dual()

    mu = 1.+s/1.e6

    return Affine(mu*IDEM + R, C)

## An example of a Helmert transform in ISCE today.
WGS84_TO_MGI = helmert(-577.326, -90.129, -463.920, -2.423, 5.137, 1.474, 5.297)
