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



"""section geo.euclid.charts

The euclid.py module is all about Tensors-- and how you add, subtract,
and multiply them. The rank-2 tensor also transforms vector--but it is
not alone. There are many ways to represent rotations in R3, and
collectively, they are known as charts on SO(3)-- the rotation group.
They live in charts.py. A nice introduction is provided here:

http://en.wikipedia.org/wiki/Charts_on_SO(3)

Versors
=======
Some people like rotation matrices. I don't. Not just because Euler
angles are always ambiguous-- it's because SO(3) -- the group of rotation
matrices--is not simply connected and you get degerenrate
coordinates (aka gimbal lock) and  other problems. Fortunatley, in any
dimension "N", the rotation group, SO(N) is cover by the group Spin(N).
But what is Spin(3) and how do you represent it? I have no idea.
Fortunately, it's isomorphic to the much more popular SU(2)....  but that
uses complex 2x2 matrices--which you have to exponentiate, and has spinors
that need to be rotated 720 degrees return to their origninal state
(what's up with that?)- it's all simply too much. The good news is H:
the quaternion group from the old days, aka the hyper-complex numbers,
will do just as well:

1, i, j ,k   (called "w", "x", "y", "z" axis)

with: i**2 = j**2 = k**2 = ijk = -1

As a group, they're a simply connected representation of rotations, are
numerically stable, can be spherical-linearaly interpolated (slerp), are
easy to represent on a computer, and in my opinion, easier to use than
matrices. (They are also the standard for on-board realtime spacecraft
control).

But wait. We only care about unit quaternions, that is, quaternions with
norm == 1. They have an uncommon name: Versors. I like it.

Still, there is always an choice in how to represent them. If you're doing
math, and don't care about rotations, the Caley-Dickson extension of the
complex numbers is best, two complex numbers and another imaginary unit
"j":

z + jz'

Hence, with:

z  =  a + ib
z' =  a'+ ib'

you get a quartet of reals:

q =  (a, ib, ja', kb').

You would think that would end it, but it isn't.

There is always a question, if you have a primitive obsession and are
using arrays to represent quaternions, no one is certain--inspite of the
unambiguous Cayley-Dickson construction--if you are doing:

(w, x, y, z) or
(x, y, z, w).

Really. No, REALLY. People put "w" last because it preserves the indexing
into the vector, while adding on a scalar. Well, I DON'T INDEX VECTORS--
they don't have items (unless they do-- A Vector of ducks quacks like 3
ducks).

I break them down into a scalar part and a vector part. The order doesn't
really matter. Hence a Versor, q,  is:

a Scalar (w)
a Vector (x,y,z).

q.w is q.scalar.w
q.x is q.vector.x and so on for y and z, and THERE IS NO q[i].

Hence:   q = Versor(scalar, vector)

Your quaternions can be singletons or numpy arrays, just like Scalar and
Vectors.

They transform with their call method (NOTE: everything transforms with
its call method):

>>>q(v)

and you compose rotations with multiplication:

q(q'(v)) = (q*q')(v) or (q'*q)(v)

Which is it, do you left or right multiply? It depends. The Versor
rotations can be converted into Matrix objects via:

q.AliasMatrix()    --> Alias transformation matrix
q.AlibiMatrix()    --> Alibi transformation matrix


Alias or Alibi?
===============
Another point of confusion is "what is a transformation?". Well, alias
transformations leave the vector unchanged and give its representation in
a different coordinate system. (Think ALIAS: same animal, different name).

Meanwhile the Alibi transformation leaves the coordinates fixed and
transforms the vector. (Think ALIBI: I wasn't there, because I was here)

Which is better? I mean this argument has been around since quatum
mechanics-- is it the Heisenberg interpretation or the Copenhagen
interpretation-- that is, are eigenstates fixed and operators evolve, or
vice versa?

NO ONE CARES: Pick one and stick with it.

Personally, I like Alibi transforms, but the GN&C community perfers Alias
transforms, so that is what I do. To transform a vector, v,  by a matrix,
M, or quaternion q:

v' = v*M
v' = (~q)*v*q


Ack, left multiplication: what a nuisance. Hence, I use the __call__
overload and do:

v' = M(v)
v' = q(v)

and the call definition along with a "compose" method is inherited from a
base class (_Alias or _Alibi). That's nice, you don't have to remember.
The base class does it for you. If you want a versor to do alibi
rotations, then make one dynamically:

AlibiVersor = type("AlibiVersor",
                  (Alibi_, Versor),
                  {"___doc__": '''Alibi transformations with a Versor'''}
                  )

Euler Angle Classes
==================
There is a base class, _EulerAngleBase, for transformations represented as
Euler angles. Like matricies and versors, you can compose and call them.
Nevertheless, there are 2 points of confusion:

(1) What are the axes
(2) What are the units.

The answer:

I don't know. No, really, the EulerAngle class doesn't know. It uses its
static class attributes:

AXES  ( a length 3 tuple of vectors reresenting ordered intrinsic
       rotations), and
Circumference    ( a float that should be 2*pi or 360)

to figure it out. So, to support common radar problems like platform
motion, there is a subclass:

YPR

which has AXES set to (z, y, x)-- so that you get a yaw rotation followed
by a pitch rotation followed by a roll rotation; Circumference=360, so
that if you have an ASCII motion file from a gyroscope, for instance, you
can do:

>>>attitude = YPR(*numpy.loadtxt("gyro.txt")

I mean "oh snap" -- it's that easy. Plus, there's a RPY class, because some
people do the rotations backwards.

Rotation Summary:
================
In the spirit of OO: you do need to know which object you have when
performing transformation. If you have "obj", then:

      obj(v)     transforms v
      ~obj       inverts the transformation
      obj*obj'   composes transformations (e.g, obj.compose(obj'))
      obj*(~obj) will yield the identitiy transformation

      obj.roll   \
      obj.yaw     > is ambigusous? This is TBD
      obj.pitch  /

      obj.AliasMatrix()     return's the equivalent matrix
      obj.versor()          return the equivalent versor
      obj.ypr()             return the equivalent YPR triplet object
      obj.rpy()             return the equivalent RPY triplet object

      obj can be a Matrix Versor YPR or RYP

      instance. Polymorphism-- don't do OO without it.
      """
##\namespace geo::charts <a href="http://en.wikipedia.org/wiki/Charts_on_SO(3)">
## Charts in SO(3)</a> for rotations.
import os
import operator
import itertools
import functools
import collections
import numpy as np
from isceobj.Util.geo import euclid

## \f$ q^{\alpha} \f$ where: \f$ \alpha > 0 \f$ \n
## <a href="http://en.wikipedia.org/wiki/Slerp">Spherical Linear Interpolation
## </a> --\n note, only interpolates between identity and versor, not 2 versors
## -- needs work, as needed.
def slerp(q, x, p=None):
    """q' = slerp(q, x)

    q, q' is a unit quaternion (Versor)
    x     is a real number (or integer).

    x = 0 --> q' = Indentity transform
        1 --> q' = q
        2 --> q' = q**2, etc, with non-integers leading to interpolation
        within the unit-hyper-sphere.
     """
    sinth = abs(q.vector).w
    theta = np.arctan2(sinth, q.scalar.w)
    rat = np.sin(x*theta)/sinth
    return Versor(euclid.Scalar(np.cos(x*theta)), q.vector*rat)

## It's a chart on SO(3), so it is here.
class Matrix(euclid.Matrix):
    pass

## Limited <a href="http://en.wikipedia.org/wiki/Versor">Versor</a> class for alias transformations.
class Versor(euclid.Geometric, euclid.Alias):
    """Versors are unit quaternions. They represent rotations. Alias
    rotations, that is rotation of coordinates, not of vectors.

    You can't add them, you can't divide them. You can:

    *   --> Grassman product
    ~   --> conjugate (inverse)
    ()  --> transform a vector argument to a representation in a new frame
    q**n  --> spherical linear interpolation (slerp)

    See __init__ for signature %s

    You can get componenets as:
    w, x, y, z, i, j, k, scalar, vector, roll, pitch, yaw

    You can get equivalent rotaiton matrices:

    q.AlibiMatrix()
    q.AliasMatrix()
    q.Matrix()    (this pick the correct one from above)

    Or tait bryan angles:

    YPR()
    """

    slots = ("scalar", "vector")

    ## \f$ {\bf q} \equiv (q; \vec{q}) \f$ \n Takes a euclid.Scalar and a
    ## euclid.Vector
    def __init__(self, scalar, vector):
        """Versor(scalar, vector):

        scalar --> sin(theta/2) as a Scalar instance
        vector --> cos(theta/2)*unit_vector as a Vector instance.

        Likewise, you can pull out:
        (w, x, y, z) if needed.
        """
        ## euclid.Scalar part
        self.scalar = scalar
        ## euclid.Vector part
        self.vector = vector
        return None

    ## Identity operation
    versor = euclid.Geometric.__pos__

    ## read-only "w" component
    @property
    def w(self):
        return self.scalar.w

    ## read-only "i" component
    @property
    def i(self):
        return self.vector.x

    ## read-only "j" component
    @property
    def j(self):
        return self.vector.y

    ## read-only "k" component
    @property
    def k(self):
        return self.vector.z

    ## \f$ ||{\bf q\cdot p }|| \equiv qp + \vec{q}\cdot\vec{p} \f$ \n
    ## The Quaternion dot product
    def inner(self, other):
        return self.scalar*other.scalar + self.vector*other.vector

    ## \f$ ||{\bf q}|| \equiv \sqrt{\bf q \cdot q} \f$
    def __abs__(self):
        return (self.inner(self))**0.5

    ## \f$ {\bf \tilde{q}} \rightarrow (q; -\vec{q}) \f$ \n Is the conjuagte,
    ## for unit quternions.
    def __invert__(self):
        """conjugate (inverse)"""
        return Versor(self.scalar, -self.vector)

    ## Grassmann() product
    def __mul__(self, versor):
        """Grassmann product"""
        return self.Grassmann(versor)

    ## Spherical Linear Interpolation (slerp())
    def __pow__(self, r):
        if r == 1:
            return self
        elif r < 0:
            return (~self)**r
        else:
            return slerp(self, r)
        pass

    def __str__(self):
        return "{"+str(self.w)+"; "+str(self.vector)+"}"

    ## \f$ {\bf q}{\bf p} = (q; \vec{q})(p; \vec{p}) = (qp-\vec{q}\cdot\vec{p}; q\vec{p} + p\vec{q} + \vec{q} \times \vec{p} ) \f$ \n
    ## Is the antisymetric product on \f$ {\bf H} \f$.
    def Grassmann(self, other):
        """Grassmann product with ANOTHER versor"""
        return self.__class__(
            self.scalar.__mul__(
                other.scalar
                ).__sub__(self.vector.__mul__(other.vector)),
            (
                self.scalar.__mul__(other.vector).__add__(
                    self.vector.__mul__(other.scalar)).__add__(
                    (self.vector).cross(other.vector)
                    )
                )
            )

    ## \f$ {\bf q}(\vec{v}) \rightarrow \vec{v}' \f$ with \n \f$ (0, \vec{v}')  = {\bf \tilde{q}(0; \vec{v})q} \f$ \n
    ## is an alias transformation by similarity transform using Grassmann()
    ## multiplication (of the versor inverse).
    def AliasTransform(self, vector):
        return (
            (~self).Grassmann(
                vector.right_quaternion().Grassmann(self)
                )
            ).vector

    ## This is the inverse of the AliasTransform
    def AlibiTransform(self, vector):
        return (~self).AliasTransform(vector)

    ## \f${\bf q}\rightarrow M=(2q^2-1)I+2(q\vec{q}\times+2\vec{q}\vec{q}) \f$
    def AlibiMatrix(self):
        """equivalent matrix for alibi rotation"""
        return  (
            (2*self.scalar**2-1.)*euclid.IDEM+
            2*(self.scalar*(self.vector.dual())+
               (self.vector.outer(self.vector))
               )
            )

    ##\f${\bf q}\rightarrow M=[(2q^2-1)I+2(q\vec{q}\times+2\vec{q}\vec{q})]^T\f$
    def AliasMatrix(self):
        """equivalent matrix for alias rotation"""
        return self.AlibiMatrix().T

    ## AliasMatrix()'s yaw
    @property
    def yaw(self):
        """Yaw angle (YPR ordering)"""
        return self.AliasMatrix().yaw

    ## AliasMatrix()'s pitch
    @property
    def pitch(self):
        """Pitch angle (YPR ordering)"""
        return self.AliasMatrix().pitch

    ## AliasMatrix()'s roll
    @property
    def roll(self):
        """Roll angle (YPR ordering)"""
        return self.AliasMatrix().roll

    ## A triplet of angles
    def ypr(self):
        """yaw, pitch, roll tuple"""
        return self.AliasMatrix().ypr()

    ## as a YPR instance
    def YPR(self):
        """YPR instance equivalent"""
        return self.AliasMatrix().YPR()

    ## A triplet of (x, y, z) in the rotated frame.
    def new_basis(self):
        """map(self, (x,y,z))"""
        return map(self, euclid.BASIS)

    ## Compute the look angles by transforming the boresite and getting is
    ## Vector.Polar polar (elevation) and azimuth angle.
    def look_angle(self, boresite=euclid.Z):
        """q.look_angle([boresite=euclid.Z])

        get a euclid.LookAngle tuple.
        """
        return self(boresite).Polar(look_only=True)

    ## \f$ x \equiv i \f$
    x=i
    ## \f$ y \equiv j \f$
    y=j
    ## \f$ z \equiv k \f$
    z=k
    pass

## A Base class for <a href="http://en.wikipedia.org/wiki/Euler_angles">Euler
## Angles</a>: it defines operations, but does not define axis order or units.
class _EulerAngleBase(euclid.Geometric, euclid.Alias):

    ## \f$ (\alpha, \beta, \gamma) \f$
    slots = ('alpha', 'beta', 'gamma')

    ## \f$ (\alpha, \beta, \gamma) \f$ -- units are unknown
    def __init__(self, alpha, beta, gamma):
        ## \f$ \alpha \f$, 1st rotation
        self.alpha = alpha
        ## \f$ \beta \f$, 2nd rotation
        self.beta  = beta
        ## \f$ \gamma \f$, 3rd rotation
        self.gamma = gamma
        return None

    ## Use Versor() --sub classes are responsible for putting it in correct
    ## form after using call super.
    def __invert__(self):
        return ~(self.versor())

    ## Use Versor()  --sub classes are responsible for putting it in correct
    ## form
    def __mul__(self, other):
        return (self.versor()*other.versor())

    def __pow__(self, *args):
        raise TypeError(
            "Euler Angler powers are not supported, use Versors and slerp"
            )

    ## rotation, counted from 0.
    def _rotation(self, n):
        return self.__class__.AXES[n].versor(
            getattr(self,
                    self.__class__.slots[n]
                    ),
            circumference=self.__class__.Circumference
            )

    ## get 1st, 2nd, or 3 rotation Versor
    def rotation(self, n):
        """versor = rotation(n) for n = 1, 2 ,3

        gets the Versor representing the n-th rotation.
        """
        return self._rotation(n-1)

    ## Compose the 3 rotations using chain().
    def versor(self):
        """Compute the equvalent Versor for all three rotations."""
        return self.chain(*map(self._rotation, range(euclid.DIMENSION)))

    ## Use Versor()   --sub classes are responsible for putting it in correct
    ## form
    def AliasMatrix(self):
        """Transformation as a Matrix"""
        return self.versor().AliasMatrix()

    ## Aliasi transformation of arguement, using versor(), but effectively: \n
    ## \f$  {\bf \vec{v}'} = {\bf \vec{v} \cdot M} \f$
    def AliasTransform(self, vector):
        """Apply transformation to argument"""
        return self.versor()(vector)

    pass

## Traditional Euler Angles
class EulerAngle(_EulerAngleBase):

    ## Intrinsic rotation
    AXES = (euclid.Z, euclid.Y, euclid.Z)

    ## In radians
    Circumference = 2*np.pi
    pass


## Tait Bryan angles are for flight dynamics.
class TaitBryanBase(_EulerAngleBase):

    ## Define angular unit (as degrees)
    Circumference = 360.

    ## \f$ \beta \f$
    @property
    def pitch(self):
        return self.beta

    pass

## Yaw Pitch Roll
class YPR(TaitBryanBase):
    """YPR(yaw, pitch, roll) --all in degrees
    and in that order, polymorphic with Versors and rotation matrices.
    """
    ## Yaw, Pitch, and *then* Roll
    AXES = (euclid.Z, euclid.Y, euclid.X)

    ## Multiplicative inverse, invokes a call super and conversion back to YPR
    def __invert__(self):
        return super(YPR, self).__invert__().YPR()

    ## Multiplication, invokes a call super and conversion back to YPR
    def __mul__(self, other):
        return super(YPR, self).__mul__(other).YPR()

    ## \f$ \alpha \f$
    @property
    def yaw(self):
        return self.alpha

    ## \f$ \gamma \f$
    @property
    def roll(self):
        return self.gamma

    pass


## Roll Pitch ROll
class RPY(TaitBryanBase):
    """RPY(roll, pitch, yaw) --all in degrees
    and in that order
    """
    ## Yaw, Pitch, and *then* Roll
    AXES = (euclid.X, euclid.Y, euclid.Z)

    ## Multiplicative inverse, invokes a call super and conversion back to YPR
    def __invert__(self):
        return super(RPY, self).__invert__().RPY()

    ## Multiplication, invokes a call super and conversion back to YPR
    def __mul__(self, other):
        return super(RPY, self).__mul__(other).RPY()

    ## \f$ \gamma \f$
    @property
    def yaw(self):
        return self.gamma

    ## \f$ \alpha \f$
    @property
    def roll(self):
        return self.alpha

    pass

## The "Real" Quaternoin Basis unit
W = Versor(euclid.ONE, euclid.NULL)
## The 3 hyper imaginary Quaternion Basis units
I, J, K = map(operator.methodcaller("versor", 1, circumference=2), euclid.BASIS)

## A private decorator for making Roll(), Pitch(), and Yaw() functions from
## axis names- this may go too far: the module functions just return the
## string name of the axis (which must be in Vector.slots), and this
## decorator goes and get that and makes a function that rotates around
## that axis--so at least its a DRY solution, if not a bit abstract.
def _flight_dynamics(func):
    """This decorator get's a named axis and makes a partial function that
    rotates about it in degrees, using the unit vector's versor() method"""
    attr = func(None)
    result = functools.partial(
        getattr(euclid.BASIS, attr).versor,
        circumference=360.
        )
    result.__doc__ = (
        """versor=%s(angle)\nVersor for alias rotation about %s axis (deg)""" %
        (str(func).split()[1], attr)
        )
    return result

## Roll coordinate transformation (in degrees) \n
## (http://en.wikipedia.org/wiki/Flight_dynamics)
@_flight_dynamics
def Roll(angle):
    return 'x'

## Pitch coordinate transformation (in degrees) \n
## (http://en.wikipedia.org/wiki/Flight_dynamics)
@_flight_dynamics
def Pitch(angle):
    return 'y'

## Yaw coordinate transformation (in degrees) \n
## (http://en.wikipedia.org/wiki/Flight_dynamics)
@_flight_dynamics
def Yaw(angle):
    return 'z'
