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



"""euclid is for geometric objects in E3.

The main objects are:
------------------------------------------------------------------
Scalar             rank-0 tensors
Vector             rank-1 tensors
Tensor (Matrix)    rank-2 tensors

   use their docstrings.

The main module constants are:
-------------------------------------------------------------------
AXES = 'xyz'     -- this names the attributes for non-trivial ranks.
BASIS            -- is a named tuple of standard basis vectors
IDEM             -- is the Idem-Tensor (aka identity matrix)

The main module functions are: really for internal use only, but they
are not protected with a leading "_".

Other:
------
There is limited support for vectors in polar coordinates. There is a:

Polar named tuple,
polar2vector conviniece constructor, and
Vecotr.polar() method

You can build Tensor (Matrix) objects from 3 Vectors using:

ziprows,  zipcols

Final Note on Classes: all tensor objects have special methods:
----------------------

slots                        These are the attributes (components)

iter()            v.iter()       Iterates over the components
tolist()                         ans puts them in a list

__getitem__       v[start:stop:step]  will pass __getitem__ down to the
                                      attributes

__iter__          iter(v)        will take array_like vectors an return
                                 singleton-like vectors as an iterator

next              next(v)        see __iter__()

mean(axis=None)    \
sum(axis=None)      > apply numpy methods to components, return tensor object.
cumsum(axis=None)  /

append:        v.append(u)  --> for array like v.x, ..., v.z; append
                                 u.x, ..., u.z onto the end.

broadcast(func,*args, **kwargs)  apply func(componenet, *args, **kwargs) for
                                 each componenet.

__contains__      u in v         test if singlton-like u is in array_like v.


__cmp__           u == v      etc., when it make sense
__nonzero__       bool(v)     check for NULL values.

These work on Scalar, Vector, Tensor, ECEF, LLH, SCH, LTP objects.

See charts.__doc__ for a dicussion of transformation definition
"""
## \namespace geo::euclid Geometric Animals living in
## <a href="http://en.wikipedia.org/wiki/Euclidean_space">\f$R^3\f$</a>

__date__ = "10/30/2012"
__version__ = "1.21"

import operator
import itertools
from functools import partial, reduce
import collections

import numpy as np

## Names of the coordinate axes
AXES = 'xyz'

## Number of Spatial Dimensions
DIMENSION = len(AXES)

## This function gets components into a list
components = operator.methodcaller("tolist")

## This function makes a generator that generates tensor components
component_generator = operator.methodcaller("iter")

## compose is a 2 arg functions that invokes the left args compose method with
## the right arg as an argument (see chain() ).
def compose(left, right):
    """compose(left, right)-->left.compose(right)"""
    return left.compose(right)

## A named tuple for polar coordinates in terms of radius, polar angle, and
## azimuth angle It has not been raised to the level of a class, yet.
Polar = collections.namedtuple("Polar", "radius theta phi")

## This is the angle portion of a Polar
LookAngles = collections.namedtuple("LookAngle", "elevation azimuth")

## get the rank from the class of the argument, or None.
def rank(tensor):
    """get rank attribute or None"""
    try:
        result = tensor.__class__.rank
    except AttributeError:
        result = None
        pass
    return result


## \f$ s = v_iv'_i \f$ \n Two Vector()'s --> Scalar .
def inner_product(u, v):
    """s = v_i v_i"""
    return Scalar(
        u.x*v.x +
        u.y*v.y +
        u.z*v.z
        )

## dot product assignemnt
dot = inner_product

## \f$ v_i = \epsilon_{ijk} v'_j v''_k \f$ \n Two Vector()'s --> Vector .
def cross_product(u, v):
    """v"_i = e_ijk v'_j v_k"""
    return u.__class__(
        u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x
        )

## cross product assignment
cross = cross_product

## \f$ m_{ij} v_iv'_j \f$ \n Two Vector()'s --> Matrix .
def outer_product(u, v):
    """m_ij = u_i v_j"""
    return Matrix(
        u.x*v.x, u.x*v.y, u.x*v.z,
        u.y*v.x, u.y*v.y, u.y*v.z,
        u.z*v.x, u.z*v.y, u.z*v.z
        )

## dyad is the outer product
dyadic = outer_product

##\f${\bf[\vec{u},\vec{v},\vec{w}]}\equiv{\bf\vec{u}\cdot(\vec{v}\times\vec{w})}\f$
## \n Three Vector()'s --> Scalar .
def scalar_triple_product(u, v, w):
    """s = v1_i e_ijk v2_j v3_k"""
    return inner_product(u, cross_product(v, w))

## \f${\bf  \vec{u} \times (\vec{v} \times \vec{w})} \f$   \n
## Three Vector()'s --> Scalar .
def vector_triple_product(u, v, w):
    """v3_i = e_ijk v1_j e_klm v1_l v2_m"""
    return reduce(operator.xor, reversed((u, v, w)))

## \f$ v'_i = m_{ij}v_j \f$ \n Matrix() times a Vector() --> Vector .
def posterior_product(m, u):
    """v'_i = m_ij v_j"""
    return u.__class__(m.xx*u.x + m.xy*u.y + m.xz*u.z,
                       m.yx*u.x + m.yy*u.y + m.yz*u.z,
                       m.zx*u.x + m.zy*u.y + m.zz*u.z)

## \f$ v'_i = m_{ji}v_j \f$ \n Vector() times a Matrix() --> Vector .
def anterior_product(v, m):
    """v'_j = v_i m_ij"""
    return v.__class__(m.xx*v.x + m.yx*v.y + m.zx*v.z,
                       m.xy*v.x + m.yy*v.y + m.zy*v.z,
                       m.xz*v.x + m.yz*v.y + m.zz*v.z)

## \f$ m_{ij} = m'_{ik}m''_{kj} \f$ \n Matrix() input and output.
def matrix_product(m, t):
    """m_ik = m'_ij m''_jk"""
    return Matrix(
        m.xx*t.xx + m.xy*t.yx + m.xz*t.zx,
        m.xx*t.xy + m.xy*t.yy + m.xz*t.zy,
        m.xx*t.xz + m.xy*t.yz + m.xz*t.zz,

        m.yx*t.xx + m.yy*t.yx + m.yz*t.zx,
        m.yx*t.xy + m.yy*t.yy + m.yz*t.zy,
        m.yx*t.xz + m.yy*t.yz + m.yz*t.zz,

        m.zx*t.xx + m.zy*t.yx + m.zz*t.zx,
        m.zx*t.xy + m.zy*t.yy + m.zz*t.zy,
        m.zx*t.xz + m.zy*t.yz + m.zz*t.zz,
        )

## Scalar times number--> Scalar .
def scalar_dilation(s, a):
    """s' = scalar_dilation(s, a)

    s, s'  is a Scalar instance
    a      is a number.
    """
    return Scalar(s.w*a)

## Vector times number --> Vector .
def vector_dilation(v, a):
     """v' = vector_dilation(v, a)

     v, v'  is a vector instance
     a      is a number.
     """
     return v.__class__(a*v.x, a*v.y, a*v.z)

## Matrix times a number --> Matrix .
def matrix_dilation(m, a):
    """m' = matrix_dilation(m, a)

    m, m'  is a Matrix instance
    a      is a number.
    """
    return Matrix(a*m.xx, a*m.xy, a*m.xz,
                  a*m.yx, a*m.yy, a*m.yz,
                  a*m.zx, a*m.zy, a*m.zz)


## Multiply 2 Scalar inputs and get a Scalar .
def scalar_times_scalar(s, t):
    """s' = scalar_dilation(s, a)

    s, s'  is a Scalar instance
    a      is a number.
    """
    return scalar_dilation(s, t.w)


## Multiply a Scalar and a Vector to get a Vector .
def scalar_times_vector(s, v):
    """v' = scalar_times_vector(s, v)

    s       is a Scalar
    v, v'   is a Vector
    """
    return vector_dilation(v, s.w)

## Multiply a Vector and a Scalar to get a Vector .
def vector_times_scalar(v, s):
    """v' = vector_times_scalar(v, s)

    s       is a Scalar
    v, v'   is a Vector
    """
    return vector_dilation(v, s.w)

## Multiply a Scalar and a Matrix to get a Matrix .
def scalar_times_matrix(s, m):
    """m' = scalar_times_matrix(s, m)

    s       is a Scalar
    m, m'   is a Matrix
    """
    return matrix_dilation(m, s.w)

## Multiply a Matrix and a Scalar to get a Matrix .
def matrix_times_scalar(m, s):
    """m' = matrix_times_scalar(m, s)

    s       is a Scalar
    m, m'   is a Matrix
    """
    return matrix_dilation(m, s.w)


## \f$ T_{ij}' = T_{kl}M_{ik}M_{jl} \f$
def rotate_tensor(t, r):
    """t' = rotate_tensor(t, r):

    t', t   rank-2 Tensor objects
    r       a rotation object.
    """
    m = r.Matrix()
    return m.T*t*m


## \f$ P \rightarrow r\sin{\theta}\cos{\phi}{\bf \hat{x}} + r\sin{\theta}\sin{\phi}{\bf \hat{y}} + r\cos{\theta}{\bf \hat{z}}  \f$
## \n Convinience constructor to convert from a Polar tuple to a Vector
def polar2vector(polar):
    """vector = polar2vector
    """
    x = (polar.radius)*np.sin(polar.theta)*np.cos(polar.phi)
    y = (polar.radius)*np.sin(polar.theta)*np.sin(polar.phi)
    z = (polar.radius)*np.cos(polar.theta)

    #  Note: if you have numpy.arrays r, theta, phi   then: BE CAREFUL with:
    #  >>>   r*Vector( sin(theta), ..., cos(theta) )
    # which gets mapped to r.__mul__ and not Vector.__rmul___
    #

    return Vector(x, y, z)



## Stack in left index (it's not a row), and it inverts Tensor.iterrows()
def ziprows(v1, v2, v3):
    """M = ziprrows(v1, v2, v3)

    stack Vector arguments into a Tensor/Matrix, M"""
    return Matrix(*itertools.chain(*map(components, (v1, v2, v3))))

## Stack in right index (it's not a column), and it inverts Tensor.itercols()
def zipcols(v1, v2, v3):
    """M = zipcols(v1, v2, v3)

    Transpose of ziprows
    """
    return ziprows(v1, v2, v3).T

## metaclass computes indices from rank and assigns them to slots\n
## This is not for users.
class ranked(type):
    """A metaclass-- used for classes with a rank static attribute that
    need slots derived from it.

    See the rank2slots static memthod"""

    ## Extend type.__new__ so that it add slots using rank2slots()
    def __new__(cls, *args, **kwargs):
        obj = type.__new__(cls, *args, **kwargs)
        obj.slots = cls.rank2slots(obj.rank)
        return obj

    ## A function that computes a tensor's attributes from it's rank\n Starting
    ## with "xyz" or "w".
    @staticmethod
    def rank2slots(rank):
        import string
        return (
            tuple(
                map(partial(string.join, sep=""),
                    itertools.product(*itertools.repeat(AXES, rank)))
                ) if rank else ('w',)
            )
    pass



## This base class controls __getitem__ behavior and provised a comonenet
## iterator.
class PolyMorphicNumpyMixIn(object):
    """This class is for classes that may have singelton on numpy array
    attributes (Vectors, Coordinates, ....).
    """

    ## Object is iterbale ONLY if its attributes are, Use with care
    def __getitem__(self, index):
        """[] --> index over components' iterator, is NOT a tensor index"""
        return self.__class__(*[item[index] for item in self.iter()])


    ## The iter() function: returns an instance that is an iterator
    def __iter__(self):
        """converts array_like comonents into iterators via iter()
        function"""
        return self.__class__(*map(iter, self.iter()))

    ## This allow you to send instance to the next function
    def next(self):
        """get's next Vector from iterator components-- you have to
        understand iterators to use this"""
        return self.__class__(*map(next, self.iter()))

    ## This allows you to use a static attribute other than "slots".
    def _attribute_list(self, attributes="slots"):
        """just an attr getter using slots"""
        return getattr(self.__class__, attributes)

    ## Note: This is JUST an iterator over components/coordinates --don't get
    ## confused.
    def iter(self, attributes="slots"):
        """return a generator that generates components """
        return (
            getattr(self, attr) for attr in self._attribute_list(attributes)
            )

    ## Matches numpy's call --note: DO NOT DEFINE a __array__ method.
    def tolist(self):
        """return a list of componenets"""
        return list(self.iter())

    ## historical assignement
    components = tolist

    ## This allows you to broadcast functions (numpy functions) to the
    ## attributes and rebuild a class
    def broadcast(self, func, *args, **kwargs):
        """vector.broadcast(func, *args, **kwargs) -->
        Vector(*map(func, vector.iter()))

        That is: apply func componenet wise, and return a new Vector
        """
        f = partial(func, *args, **kwargs)
        return self.__class__(*map(f, self.iter()))

    ## \f$ T_{i...k} \rightarrow \frac{1}{n}\sum_0^{n-1}{T_{i...k}[n]} \f$
    def mean(self, *args, **kwargs):
        """broadcast numpy.mean (see broadcast.__doc__)"""
        return self.broadcast(np.mean, *args, **kwargs)

    ## \f$ T_{i...k} \rightarrow \sum_0^{n-1}{T_{i...k}[n]} \f$
    def sum(self, *args, **kwargs):
        """broadcast numpy.sum (see broadcast.__doc__)"""
        return self.broadcast(np.sum, *args, **kwargs)

    ## \f$ T_{i...k}[n] \rightarrow \sum_0^{n-1}{T_{i...k}[n]} \f$
    def cumsum(self, *args, **kwargs):
        """broadcast numpy.cumsum (see broadcast.__doc__)"""
        return self.broadcast(np.cumsum, *args, **kwargs)

    ## experimantal method called-usage is tbd
    def numpy_method(self, method_name):
        """numpy.method(method_name)

        broadcast numpy.ndarray method (see broadcast.__doc__)"""
        return self.broadcast(operator.methodcaller(method_name))

    ## For a tensor, chart, or coordinate made of array_like objects:\n append
    ## a like-wise object onto the end
    def append(self, other):
        """For array_like attributes, append and object into the end
        """
        result = self.__class__(
            *[np.append(s, o) for s, o in zip(self.iter(), other.iter())]
             )
        if hasattr(self, "peg_point") and self.peg_point is not None:
            result.peg_point = self.peg_point
            pass
        self = result
        return result

    ## The idea is to check equality for all tensor/coordinate types and for
    ## singleton and numpy arrays, \n so this is pretty concise-- the previous
    ## versior was 10 if-then blocks deep.
    def __eq__(self, other):
        """ check tensor equality, for each componenet """
        if isinstance(other, self.__class__):
            for n, item in enumerate(zip(self.iter(), other.iter())):
                result = result * item[0]==item[1] if n else item[0]==item[1]
                pass
            return result
        else:
            return False
        pass


    ## not __eq__, while perserving array_like behavior- not easy -- note that
    ## function/statement calling enforces type checking (no array allowed), \n
    ## while method calling does not.
    def __ne__(self, other):
        inv = self.__eq__(other)
        try:
            result =  operator.not_(inv)
        except ValueError:
            result = (1-inv).astype(bool)
            pass
        return result

    ## This covers <   <=   >  >=  and raises a RuntimeError, unless numpy
    ## calls it, and then that issue is addressed (and it's complicated)
    def __cmp__(self, other):
        """This method is called in 2 cases:

        Vector Comparison: if you compare (<, >, <=, >=) two non-scalar
                           tensors, or rotations--that makes no sense and you
                           get a TypeError.

        Left-Multiply with Numpy: this is a little more subtle. If you are
                           working with array_like tensors, and say, do

        >>> (v**2).w*v   instead of:
        >>> (v**2)*v     (which works), numpy will take over and call __cmp__
                          inorder to figure how to do linear algebra on
                          the array_like componenets of your tensor. The whole
                          point of the Scalar class is to avoid this pitfall.

        Side Note: any vector operation should be manifestly covariant-- that
        is-- you don't need to access the "w"--so you should not get this error.

        But you might.
        """
        raise TypeError(
            """comparision operation not permitted on %s

    Check for left mul by a numpy array.

    Right operaned is %s""" % (self.__class__.__name__, other.__class__.__name)
            )

    ## In principle, we always want true division: this is here in case a
    ## client calls it
    def __truediv__(self, other):
        return self.__div__(other)

    ## In principle, we always want true division: this is here in case a
    ## client calls it
    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    ## This is risky and pointless-- who calls float?
    def __float__(self):
        return abs(self).w

    ## +T <a href="http://docs.python.org/reference/expressions.html#is">is</a>
    ##  T --- is as in python\n This method is used in coordinate transforms
    ## when an identity transform is required.
    def __pos__(self):
        return self

    ## For container attributes: search, otherise return NotImplemented so
    ## that the object doesn't look like a container when intropspected.
    def __contains__(self, other):
        try:
            for item in self:
                if item == other:
                    return True
                pass
            pass
        except TypeError:
            raise NotImplementedError
        pass

    ## A tensor/chart/coordinate object has a len() iff all its components
    ## have all the same length
    def __len__(self):
        for n, item in enumerate(self.iter()):
            if n:
                if len(item) != last:
                    raise ValueError(
                        "Length of %s object is ill defined" %
                        self.__class__.__name
                        )
                pass
            else:
                try:
                    last = len(item)
                except TypeError:
                    raise TypeError(
                        "%s object doesn't have a len()" %
                        self.__class__.__name__
                        )
                pass
            pass
        return last
    pass

## A temporaty class until I decide if ops are composed or convolved (brain
## freeze).
class _LinearMap(object):
    """This class ensures linear map's compose method calls the colvolve
    method"""

    ## compose is convolve
    def compose(self, other):
        return self.convolve(other)

    ## convolve is compose
    def convolve(self, other):
        return self.compose(other)

    ## Chain together a serious of instances -- it's a static method
    @staticmethod
    def chain(*args):
        """chain(*args)-->reduce(compose, args)"""
        return reduce(compose, args)

    pass


## Mixin for Alias transformations rotate the coordinate system, leaving the
## object fixed
class Alias(_LinearMap):
    """This mix-in class makes the rotation object a Alias transform:

    It rotates coordinate systems, leaving the vector unchanged.
    """
    ## As a linear map, this is __call__
    def __call__(self, other):
        return self.AliasTransform(other)

    ## \f$ (g(f(v))) = (fg)(v) \f$
    def compose(self, other):
        return self*other

    ## This is an Alias Transform
    def AliasMatrix(self):
        return self

    ## Alibi Transform is the transpose
    def AlibiMatrix(self):
        return self.T

    ## Alias.Matrix is AliasMatrix
    def Matrix(self):
        """get equivalent Alias rotatation Matrix for rotation (chart) object"""
        return self.AliasMatrix()

    pass


## Mixin Alibi transformations rotate the object, leaving the coordinare
## system fixed
class Alibi(_LinearMap):
    """This mix-in class makes the rotation object a Alibi transform:

    It rotates vectors with fixed coordinate systems.
    """
    ## As a linear map, this is __call__
    def __call__(self, other):
        return self.AlibiTransform(other)

    ## \f$ (g(f(v))) = (gf)(v) \f$
    def compose(self, other):
        return other*self

    ## This is an Alibi Transform
    def AlibiMatrix(self):
        return self

    ## Alias Transform is the transpose
    def AliasMatrix(self):
        return self.T

    ## Alibi.Matrix is AlibiMatrix
    def Matrix(self):
        """get equivalent Alibi rotatation Matrix for rotation (chart) object"""
        return self.AlibiMatrix()

    pass


## Base class for animals living in \f$ R^3 \f$.
class Geometric(PolyMorphicNumpyMixIn):
    """Base class for things that are:

    iterables over their slots

    may or may not have iterbable attributes
    """
    ## neg is the same class with all components negated, use map and
    ## operator.neg with iter().
    def __neg__(self):
        """-tensor maps components to neg and builds a new tensor.__class__"""
        return self.__class__(*map(operator.neg, self.iter()))

    ## Repr is as repr does
    def __repr__(self):
        guts = ",".join(map(str, self.iter()))
        return repr(self.__class__).lstrip("<class '").rstrip("'>")+"("+guts+")"

    ## Sometimes I just want a list of componenets
    def components(self):
        """Return a list of slots attributes"""
        return super(Geometric, self).tolist()

    pass

## This decorator decorates element-wise operations on tensors
def elementwise(op):
    """func = elementwise(op):

    op is a binary arithmetic operator.
    So is func, except it works on elements of the Tensor, returning a new
    tensor of the same type.
    """
    from functools import wraps
    @wraps(op)
    def wrapped_op(self, other):
        try:
            result =  self.__class__(
                *[op(*items) for items in itertools.zip_longest(self.iter(),
                                                                 other.iter())]
                 )
        except (TypeError, AttributeError) as err:
            from isceobj.Util.geo.exceptions import (
                NonCovariantOperation, error_message
                )
            x = (
                NonCovariantOperation if isinstance(other,
                                                    PolyMorphicNumpyMixIn)
                else
                TypeError
                )
            raise x(error_message(op, self, other))
        return result
    return wrapped_op

## Base class
class Tensor_(Geometric):
    """Base class For Any Rank Tensor"""

    ## Get the rank of the other, and chose function from the mul_rule
    ## dictionary static attribute
    def __mul__(self, other):
        """Note to user: __mul__ is inherited for Tensor. self's mul_rule
        dictionary is keyed by other's rank inorder to get the correct function
        to multiply the 2 objects.
        """
        return self.__class__.mul_rule[rank(other)](self, other)

    ## reflected mul is always a non-Tensor, so use the [None] value from the
    ## mul_rule dictionary
    def __rmul__(self, other):
        """rmul always selects self.__class__.mul_rule[None] to compute"""
        return self.__class__.mul_rule[None](self, other)

    ## elementwise() decorated addition
    @elementwise
    def __add__(self, other):
        """t3_i...k = t1_i...k + t2_i...k  (elementwise add only)"""
        return operator.add(self, other)

    ## elementwise() decorated addition
    @elementwise
    def __sub__(self, other):
        """t3_i...k = t1_i...k - t2_i...k  (elementwise sub only)"""
        return operator.sub(self, other)

    ## Division is pretty straigt forward
    def __div__(self, other):
        return self.__class__(*[item/other for item in self.iter()])

    def __str__(self):
        return reduce(operator.add,
                      map(lambda i: str(i)+"="+str(getattr(self, i))+"\n",
                          self.__class__.slots)
                      )

    ## Reduce list of squared components- you can't use sum here if the
    ## components are basic.Arrays.
    def normsq(self, func=operator.add):
        return Scalar(
            reduce(
                func,
                [item*item for item in self.iter()]
                )
            )

    ## The L2 norm is a Scalar, from normsq()**0.5
    def L2norm(self):
        """normsq()**0.5"""
        return self.normsq()**0.5

    ## The unit vector
    def hat(self):
        """v.hat() is v's unit vector"""
        return self/(self.L2norm().w)

    ## Abs value is usually the L2-norm, though it might be the determiant for
    ## rank 2 objects.
    __abs__ = L2norm

    ## See __mul__ for dilation - this may be deprecated
    def dilation(self, other):
        """v.dilation(c) --> c*v  with c a real number."""
        return self.__class__.rank[None](self, other)

    ## Return True/False for singleton-like tensors, and bool arrays for
    ## array_like input.
    def __nonzero__(self):
        try:
            for item in self:
                break
            pass
        except TypeError:
            # deal with singelton tensor/coordiantes
            return any(map(bool, self.iter()))

        # Now deal with numpy array)like attributes
        return np.array(map(bool, self))

    pass


## a decorator for <a href="http://docs.python.org/reference/datamodel.html?highlight=__cmp__#object.__cmp__">__cmp__ operators</a>
## to try scalars, and then do numbers, works for singeltons and numpy.ndarrays.
def cmpdec(func):
    def cmp(self, other):
        try:
            result = func(self.w, other.w)
        except AttributeError:
            result = func(self.w, other)
            pass
        return result
    cmp.__doc__ = "a <op> b ==> a.w <op> b.w or a.w <op> b"
    return cmp

## A decorator: "w" not "w" -- the purpose is to help scalar operatorions with
## numpy.arrays--it seems to be impossible to cover every case
def wnotw(func):
    """elementwise should decorate just fine, but it fails if you add
    an plain nd array on the right -- should that be allowed?--it is if you
    decorate with this.
    """
    def wfunc(self, right):
        """operator with Scalar checking"""
        try:
            result = (
                func(self.w, right) if rank(right) is None else
                func(self.w, right.w)
                )
        except AttributeError:
            from isceobj.Util.geo.exceptions import NonCovariantOperation
            from isceobj.Util.geo.exceptions import error_message
            raise NonCovariantOperation(error_message(func, self, right))
        return Scalar(result)
    return wfunc

## <a href="http://en.wikipedia.org/wiki/Scalar_(physics)">Scalar</a>
## class transforms as \f$ s' = s \f$
class Scalar(Tensor_):
    """s = Scalar(w)  is a rank-0 tensor with one attribute:

    s.w

    which can be a signleton, array_like, or an iterator. You need Scalars
    because they now about Vector/Tensor operations, while singletons and
    numpy.ndarrays do not.


    ZERO
    ONE

    are module constants that are scalars.
    """
    ## The ranked meta class figures out the indices
#    __metaclass__ = ranked
    slots = ('w',)

    ## Tensor rank
    rank = 0

    ## The "rule" choses the multiply function accordinge to rank
    mul_rule = {
        None:scalar_dilation,
        0:scalar_times_scalar,
        1:scalar_times_vector,
        2:scalar_times_matrix
        }

    ## explicity __init__ is just to be nice \n (and it checks for nested
    ## Scalars-- which should not happen).
    def __init__(self, w):
        ## "w" is the name of the scalar "axis"
        self.w = w.w if isinstance(w, Scalar) else w
        return None

    ## This is  a problem--it's not polymorphic- Scalars are a pain.
    def __div__(self, other):
        try:
            result = super(Scalar, self).__div__(other)
        except (TypeError, AttributeError):
            try:
                result = super(Scalar, self).__div__(other.w)
            except AttributeError:
                from isceobj.Util.geo.exceptions import (
                    UndefinedGeometricOperation, error_message
                    )
                raise UndefinedGeometricOperation(
                    error_message(self.__class__.__div__, self, other)
                    )
            pass
        return result

    ## note: rdiv does not perserve type.
    def __rdiv__(self, other):
        return other/(self.w)

#    ## 1/s need to be defined. Do not go here with numpy arrays.
#    def __rdiv__(self, other):
#        return self**(-1)*other

    @wnotw
    def __sub__(self, other):
        return operator.sub(self, other)

    @wnotw
    def __add__(self, other):
        return operator.add(self, other)

    ## pow is pretty regular
    def __pow__(self, other):
        try:
            result = (self.w)**other
        except TypeError:
            result = (self.w)**(other.w)
            pass
        return self.__class__(result)

    ## reflected pow is required, e.g: \f$ e^{i \vec{k}\vec{r}} \f$ is a Scalar
    ## in the exponent.
    def __rpow__(self, other):
        return other**(self.w)

    ## <a href="http://docs.python.org/library/operator.html"> < </a>
    ## decorated with cmpdec() .
    @cmpdec
    def __lt__(self, other):
        return operator.lt(self, other)

    ## <a href="http://docs.python.org/library/operator.html"> <= </a>
    ## decorated with cmpdec() .
    @cmpdec
    def __le__(self, other):
        return operator.le(self, other)

    ## <a href="http://docs.python.org/library/operator.html"> > </a>
    ## decorated with cmpdec() .
    @cmpdec
    def __gt__(self, other):
        return operator.gt(self, other)

    ## <a href="http://docs.python.org/library/operator.html"> >= </a>
    ## decorated with cmpdec() .
    @cmpdec
    def __ge__(self, other):
        return operator.ge(self, other)


    pass

## Scalar Null
ZERO = Scalar(0.)
## Scalar Unit
ONE = Scalar(1.)


## <a href="http://en.wikipedia.org/wiki/Vector_(physics)">Vector</a>
## class transforms as \f$ v_i' = M_{ij}v_j \f$
class Vector(Tensor_):
    """v = Vector(x, y, z) is a vector with 3 attributes:

    v.x, v.y, v.z

    Vector operations are overloaded:

    "*"  does dilation by a scalar, dot product, matrix multiply for
         for rank 0, 1, 2  objects.

    For vector arguments, u:

    v*u --> v.dot(u)
    v^u --> v.cross(u)
    v&u --> v.outer(u)

    The methods cover all the manifestly covariant equation you can write down.


    abs(v)     --> a scalar
    abs(v).w  --> a regular number or array

    v.hat()   is the unit vector along v.
    v.versor(angle)  makes a versor that rotates around v by angle.
    v.Polar()        makes a Polar object out of it.
    ~v --> v.dual()  makes a matrix that does a cross product with v.
    v.right_quaterion() makes a right quaternion: q = (0, v)
    v.right_versor()    makes a right verosr:     q = (0, v.hat())

    v.var()            makes the covariance matrix of an array_like vector.
    """

    ## the metaclass copmutes the attributes from rank
#    __metaclass__ = ranked
    slots = ('x', 'y', 'z')

    ## Vectors are rank 1
    rank = 1

    ## The hash table assigns multiplication
    mul_rule = {
        None:vector_dilation,
        0:vector_times_scalar,
        1:inner_product,
        2:anterior_product
        }

    ## init is defined explicity, eventhough the metaclass can do it implicitly
    def __init__(self, x, y, z):
        ## x compnent
        self.x = x
        ## y compnent
        self.y = y
        ## z compnent
        self.z = z
        return None

    def __str__(self):
        return str(self.components())

    ## ~v --> v.dual()   See dual() , I couldn't resist.
    def __invert__(self):
        return self.dual()

    ## u^v --> cross_product() \n An irrestistable overload, given then wedge
    ## product on exterior algebrea-- watchout for presednece rules with this
    ## one.
    def __xor__(self, other):
        return cross_product(self, other)

    ## u&v --> outer_product() \n Wanton overloading with bad presednece rules,
    ## \n but it is the only operation that preserves everything about the
    ## arguments (syntactically AND).
    def __and__(self, other):
        return outer_product(self, other)

    ## Ths is a limited "pow"-- don't do silly exponents.
    def __pow__(self, n):
        try:
            result= reduce(operator.mul, itertools.repeat(self, n))
        except TypeError as err:
            if isinstance(n, Geometric):
                from isceobj.Util.geo.exceptions import (
                    UndefinedGeometricOperation, error_message
                    )
                raise UndefinedGeometricOperation(
                    error_message(self.__class__.__pow__, self, n)
                    )
            if n <= 1:
                raise ValueError("Vector exponent must be 1,2,3...,")
            raise err
        return result

    ## \f$ v_i u_j \f$ \n The Scalar, inner, dot product
    def dot(self, other):
        """scalar product"""
        return inner_product(self, other)

    ## \f$ c_{i} = \epsilon_{ijk}a_jb_k \f$ \n The (pseudo)Vector wedge, cross
    ## product
    def cross(self, other):
        """cross product"""
        return cross_product(self, other)

    ## \f$ m_{ij} = v_i u_j \f$ \n The dyadic, outer product
    def dyad(self, other):
        """Dyadic product"""
        return outer_product(self, other)

    outer = dyad

    ## Define a rotation about \f$ \hat{v} \f$ \n, realitve to kwarg:
    ## circumference = \f$2\pi\f$
    def versor(self, angle, circumference=2*np.pi):
        """vector(angle, circumfrence=2*pi)

        return a unit quaternion (versor) that represents an
        alias rotation by angle about vector.hat()
        """
        from isceobj.Util.geo.charts import Versor
        f = 2*np.pi/circumference
        return Versor(
            Scalar(self._ones_like(np.cos(f*angle/2.))),
            self.hat()*(np.sin(f*angle/2.))
            )

    ## Convert to a <a href="http://en.wikipedia.org/wiki/Classical_Hamiltonian_quaternions#Right_versor">right versor</a> after normalization.
    def right_versor(self):
        return self.hat().right_quaternion()

    ## Convert to a <a href="http://en.wikipedia.org/wiki/Classical_Hamiltonian_quaternions#Right_quaternion">right versor</a> (for transformation)\n That is: as add a ::ZERO Scalar part and don't normalize to unit hypr-sphere.
    def right_quaternion(self):
        from isceobj.Util.geo.charts import Versor
        return  Versor(Scalar(self._ones_like(0.)), self)

    ## \f$ v_i \rightarrow \frac{1}{2} v_i \epsilon_{ijk} \f$ \n
    ## This method is used when converting a Versor to a rotation Matrix \n
    ## it's more of a cross_product partial function operator than a Hodge dual.
    def dual(self):
        """convert to antisymetrix matrix"""
        zero = self._ones_like(0)
        return Matrix(zero,  self.z, -self.y,
                      -self.z, zero,  self.x,
                      self.y, -self.x,  zero)

    ## \f$ {\bf P} = \hat{v}\hat{v} \f$ \n
    ## <a href=" http://en.wikipedia.org/wiki/Projection_(linear_algebra)\#Orthogonal_projections ">The Projection Operator</a>: Matrix for the orthogonal projection onto vector .
    def ProjectionOperator(self):
        """vector --> matrix that projects (via right mul) argument onto vector"""
        u = self.hat()
        return u&u

    ## \f$ \hat{v}(\hat{v}\cdot\vec{u}) \f$ \n
    ## Apply ProjectionOperator() to argument.
    def project_other_onto(self, other):
        return self.ProjectionOperator()*other

    ## \f$ {\bf R} = {\bf I} - 2\hat{v}\hat{v} \f$ \n
    ## Matrix reflecting vector about plane perpendicular to vector .
    def ReflectionOperator(self):
        """vector --> matrix that reflects argument about vector"""
        return IDEM - 2*self.ProjectionOperator()

    ##  \f$ \vec{u} - \hat{v}(\hat{v}\cdot\vec{u}) \f$ \n
    ## Apply RelectionOperatior() to argument
    def reflect_other_about_orthogonal_plane(self, other):
        return self.ReflectionOperator()*other

    ## \f$ \vec{a}\cdot\hat{b} \f$ \n
    ## Scalar projection: a's projection onto b's unit vector.
    def ScalarProjection(self, other):
        """Scalar Projection onto another vector"""
        return self*(other.hat())

    ## \f$ (\vec{a}\cdot\hat{b})\hat{b} \f$ \n
    ## Vector projection: a's projection onto b's unit vector times b's unit
    ## vector\n (aka: a's resolute on b).
    def VectorProjection(self, other):
        """Vector Projection onto another vector"""
        return self.ScalarProjection(other)*(other.hat())

    ## Same thing, different name
    Resolute = VectorProjection

    ## \f$ \vec{a} - (\vec{a}\cdot\hat{b})\hat{b} \f$ \n Vector rejection
    ## (perpto) is the vector part of a orthogonal to its resolute on b.
    def VectorRejection(self, other):
        """Vector Rejection of another vector"""
        return self-(self.VectorProjection(other))

    ## \f$ \cos^{-1}{\hat{a}\cdot\hat{b}} \f$ as a Scalar instance \n
    ## this farms out the trig call so the developer doesn't have to worry
    ## about it.
    def theta(self, other):
        """a.theta(b) --> Scalar( a*b / |a||b|) --for real, it's a rank-0 obj"""
        return (self.hat()*other.hat()).broadcast(np.acos)

    ## convert to  ::Polar named tuple-- can make polar a class if needed.
    def Polar(self, look_only=False):
        """Convert to polar coordinates"""
        radius = abs(self).w
        theta  = np.arccos(self.z/radius)
        phi    = np.arctan2(self.y, self.x)
        return (
            LookAngles(theta, phi) if look_only else
            Polar(radius, theta, phi)
            )

    ##\f$\bar{(\vec{v}-\langle\bar{v}\rangle)(\vec{v}-\langle\bar{v}\rangle)}\f$
    ## For an iterable Vector
    def var(self):
        """For an iterable vector, return a covariance matrix"""
        v = (self - self.mean())
        return (v&v).mean()

    ## Get like wise zeros to fill in matrices and quaternions\n-
    ## this may not be the best way
    def _ones_like(self, constant=1):
        return constant + self.x*0

    ## Make a Gaussian Random Vector
    @classmethod
    def grv(cls, n):
        """This class method does:

        Vector(*[item(n) for item in itertools.repeat(np.random.randn, 3)])

        get it? That's a random vector.
        """
        return cls(
            *[item(n) for item in itertools.repeat(np.random.randn, 3)]
            )

    ## return self-- for polymorphism
    def vector(self):
        return self

    ## Upgrayed to a ::motion::SpaceCurve().
    def space_curve(self, t=None):
        """For array_like vectors, make a full fledged motion.SpaceCurve:

        space_curve = vector.space_curve([t=None])
        """
        from isceobj.Util.geo.motion import SpaceCurve
        return SpaceCurve(*self.iter(), t=t)

    pass

## \f$ \hat{e}_x, \hat{e}_y, \hat{e}_z \f$ \n
## The <a href="http://en.wikipedia.org/wiki/Standard_basis">standard basis</a>,
## as a class attribute</a> Redundant with module ::BASIS --
Vector.e = collections.namedtuple("standard_basis", "x y z")(
    Vector(1.,0.,0.),
    Vector(0.,1.,0.),
    Vector(0.,0.,1.)
    )

## Limited <a href="http://en.wikipedia.org/wiki/Tensor">Rank-2 Tensor</a>
## class transforms as \f$ T_{ij}' = M_{ik}M_{jl}T_{jl} \f$
class Tensor(Tensor_, Alias):
    """T = Tensor(
    xx, xy, xz,
    yx, yy, yz,
    zx, zy, zz
    )

    Is a cartesian tensor, and it's a function from E3-->E3 (a rotation matrix).

    As a rotation, it does either Alias or Alibi rotation-- it depends which
    class is in Tensor.__bases__[-1] -- it should be Alias, so it rotates
    coordinates and leaves vectors fixed.

    TRANSFORMING VECTORS:

    >>>v_prime = T(v)

    That is, the __call__ method does it for you. Use it-- do not multiply. You
    can multiply or do an explicit transformation with any of the following:

    T.AliasTransfomation(v)
    T.AlibiTransfomation(v)
    T*v
    v*T

    You can convert to other charts on SO(3) with:

    T.versor()
    T.YPR()

    Or get individual angles:

    T.yaw, T.pitch, T.roll

    Other matrix/tensor methods are:

    T.T       (same as T.transpose())
    ~T        (same as T.I inverts it)
    abs(T)    (same as T.det(), a Scalar)
    T.trace()    trace (a Scalar)
    T.L2norm()   L2-norm ( a Scalar)
    T.A()      antisymmetric part
    T.S()      symmetric part
    T.stf()    symmetric trace free part
    T.dual()   Antisymetric part (contracted with Levi-Civita tensor)

    Finally:

    T.row(n), T.col(n), T.iterrows(), T.itercols(), T.tolist() are all

    pretty simple.
    """
    ## The meta class computes the attributes from ranked
#    __metaclass__ = ranked
    slots = ('xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz')

    ## The rank is 2.
    rank = 2

    ## The hash table assigns multiplication
    mul_rule = {
        None:matrix_dilation,
        0:matrix_times_scalar,
        1:posterior_product,
        2:matrix_product
        }

    ## self.__class__.__bases__[2] rules for call, usage is TBD
    call_rule = {
        None: lambda x:x,
        0: lambda x:x,
        1: "vector_transform_by_matrix",
        2: "tensor_transform_by_matrix"
        }

    ## explicit 9 argument init.
    def __init__(self, xx, xy, xz, yx, yy, yz, zx, zy, zz):
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{xx} = {\bf{T}}^{({\bf e_x})}_x \f$
        self.xx = xx
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{xy} = {\bf{T}}^{({\bf e_y})}_x \f$
        self.xy = xy
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{xz} = {\bf{T}}^{({\bf e_z})}_x \f$
        self.xz = xz
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{yx} = {\bf{T}}^{({\bf e_x})}_y \f$
        self.yx = yx
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{yy} = {\bf{T}}^{({\bf e_y})}_y \f$
        self.yy = yy
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{yz} = {\bf{T}}^{({\bf e_z})}_y \f$
        self.yz = yz
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{zx} = {\bf{T}}^{({\bf e_x})}_z \f$
        self.zx = zx
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{zy} = {\bf{T}}^{({\bf e_y})}_z \f$
        self.zy = zy
        ## <a href="http://en.wikipedia.org/wiki/Tensor">Cartesian Componenet:
        ## </a> \f$ m_{zz} = {\bf{T}}^{({\bf e_z})}_z \f$
        self.zz = zz
        return None

    ## Alibi transforms are from the left
    def AlibiTransform(self, other):
        return posterior_product(self, other)

    ## Alias transforms are from the right
    def AliasTransform(self, other):
        return anterior_product(other, self)

    ## \f$ v_i = m_{ni} \f$ \n Get a "row", or, run over the 1st index
    def row(self, n):
        """M.row(n) --> Vector(M.nx, M.ny, M.nz)

        for n = (0,1,2) --> (x, y, z), so it's not
        a row, but a run on the 2nd index
        """
        return Vector(
            *[getattr(self, attr) for attr in
             self.slots[n*DIMENSION:(n+1)*DIMENSION]
              ]
             )

    ## \f$ v_i = m_{in} \f$
    def col(self, n):
        """Run on 1st index. See row.__doc__ """
        return self.T.row(n)

    ## iterate of rows
    def iterrows(self):
        """iterator over row(n) for n in 0,1,2 """
        return map(self.row, range(DIMENSION))

    ## make a list
    def tolist(self):
        """A list of components -- nested"""
        return [item.tolist() for item in self.iterrows()]

    ## \f$ m_{ij}^T = m_{ji} \f$
    def transpose(self):
        """Transpose: M_ij --> M_ji """
        return Matrix(self.xx, self.yx, self.zx,
                      self.xy, self.yy, self.zy,
                      self.xz, self.yz, self.zz)

    ## assign "T" to transpose()
    @property
    def T(self):
        return self.transpose()

    ## ~Matrix --> Matrix.I
    def __invert__(self):
        return self.I

    ## Matrix Inversion as a property to look like numpy
    @property
    def I(self):
        row0, row1, row2 = self.iterrows()
        return zipcols(
            cross(row1, row2),
            cross(row2, row0),
            cross(row0, row1)
            )/self.det().w

    ## \f$ m_{ii} \f$ \n Trace, is a Scalar
    def trace(self):
        return Scalar(self.xx + self.yy + self.zz)

    ## \f$ v_k \rightarrow \frac{1}{2} m_ij \epsilon_{ijk} \f$ \n
    ## Rank-1 part of Tensor
    def vector(self):
        """Convert to a vector w/o scaling"""
        return Vector(self.yz-self.zy,
                      self.zx-self.xz,
                      self.xy-self.yx)

    ## \f$ \frac{1}{2}[(m_{yz}-m_{zy}){\bf \hat{x}} + (m_{zx}-m_{xz}){\bf \hat{y}} + (m_{zx}-m_{xz}){\bf \hat{z}})] \f$ --normalization is under debate.
    def dual(self):
        """The dual is a vector"""
        return self.vector()/2.

    ## \f$ \frac{1}{2}(m_{ij} + m_{ji}) \f$ \n Symmetric part
    def S(self):
        """Symmeytic Part"""
        return (self + self.T)/2

    ## \f$ \frac{1}{2}(m_{ij} - m_{ji}) \f$ \n Antisymmetric part
    def A(self):
        """Antisymmeytic Part"""
        return (self - self.T)/2

    ## \f$ \frac{1}{2}(m_{ij} + m_{ji}) -\frac{1}{3}\delta_{ij}Tr{\bf m} \f$\n
    ## Symmetric Trace Free part
    def stf(self):
        """symmetric trace free part"""
        return self.S() - IDEM*self.trace()/3.

    ## Determinant as a scalar_triple_product as:\n
    ## \f$ m_{ix} (m_{jy}m_{kz}\epsilon_{ijk}) \f$.
    def det(self):
        """determinant as a Scalar"""
        return scalar_triple_product(*self.iterrows())

    ## |M| is determinant--though it may be negative
    __abs__ = det

    ## not quite right
    def __str__(self):
        return "\n".join(map(str, self.iterrows()))

    ## does not enfore integer only, though non-integer is not supported
    def __pow__(self, n):
        if n < 0:
            return self.I.__pow__(-n)
        else:
            return reduce(operator.mul, itertools.repeat(self, n))
        pass

    @property
    ## Get Yaw Angle  (\f$ \alpha \f$ )  as a rotation (norm is NOT checked)
    ## via: \n \f$ \tan{\alpha} = \frac{M_{yx}}{M_{xx}} \f$
    def yaw(self):
        from numpy import arctan2, degrees
        return degrees(arctan2(self.yx, self.xx))


    ## Get Pitch Angle (\f$ \beta \f$ ) as a rotation (norm is NOT checked)
    ## via: \n
    ##\f$\tan{\beta}=\frac{M_{zy}}{(M_{zy}+M_{zz})/(\cos{\gamma}+\sin{\gamma})}\f$
    def _pitch(self, roll=None):
        from numpy import arctan2, degrees, radians, cos, sin
        roll = radians(self.roll) if roll is None else radians(roll)
        cos_b = (self.zy + self.zz)/(cos(roll)+sin(roll))
        return degrees(arctan2(-(self.zx), cos_b))

    ## Use _pitch()
    @property
    def pitch(self):
        return self._pitch()

    @property
    ## Get Roll Angle  (\f$ \gamma \f$ )  as a rotation (norm is NOT checked)
    ## via: \n \f$ \tan{\gamma} = \frac{M_{zy}}{M_{zx}} \f$
    def roll(self):
        from numpy import arctan2, degrees
        return degrees(arctan2(self.zy, self.zz))

    ## Convert to a tuple of ( Yaw(), Pitch(), Roll() )
    def ypr(self):
        """compute to angle triplet"""
        roll  = self.roll
        pitch = self._pitch(roll=roll)
        return (self.yaw, pitch, roll)

    ## Convert to a YPR instance via ypr()
    def YPR(self):
        """convert to YPR class"""
        from isceobj.Util.geo.charts import YPR
        return YPR(*(self.ypr()))

    def rpy(self):
        return NotImplemented

    RPY = rpy

    ## Convert to a rotation versor via YPR()
    def versor(self):
        """Convert to a rotation versor--w/o checking for
        viability.
        """
        return self.YPR().versor()

    pass

## Synonym-- really should inherit from a biinear map and a tensor
Matrix = Tensor

## The NULL vector
NULL = Vector(0.,0.,0.)

## Idem tensor
IDEM = Tensor(
    1.,0.,0.,
    0.,1.,0.,
    0.,0.,1.
    )

## \f$ R^3 \f$ basis vectors
BASIS = collections.namedtuple("Basis", 'x y z')(*IDEM.iterrows())

## The Tuple of Basis Vectors (is here, because EulerAngleBase needs it)
X, Y, Z = BASIS

## A collections for orthonormal dyads.
DYADS = collections.namedtuple("Dyads", 'xx xy xz yx yy yz zx zy zz')(*[left&right for left in BASIS for right in BASIS])
