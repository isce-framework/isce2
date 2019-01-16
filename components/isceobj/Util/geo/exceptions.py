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



"""Some specialized arithmetic exceptions for
Vector and Affine Spaces.
"""
## \namespace geo::exceptions
## <a href="http://docs.python.org/2/library/exceptions.html">Exceptions</a>
## for Vector and Affines spaces.

## Base class for geometric errors
class GeometricException(ArithmeticError):
    """A base class- not to be raised"""
    pass

## A reminder to treat geometric objects properly.
class NonCovariantOperation(GeometricException):
    """Raise when you do something that is silly[1], like adding
    a Scalar to a Vector\.
    [1]Silly: (adj.) syn: non-covariant"""
    pass

## A reminder that Affine space are affine, and vector spaces are not.
class AffineSpaceError(GeometricException):
    """Raised when you forget the points in an affine space are
    not vector in a vector space, and visa versa"""
    pass

## A catch-all for overlaoded operations getting non-sense.
class UndefinedGeometricOperation(GeometricException):
    """This will raised if you get do an opeation that has been defined for
    a Tensor/Affine/Coordinate argument, but you just have a non-sense
    combinabtion, like vector**vector.
    """
    pass


## This function should make a generic error message
def error_message(op, left, right):
    """message = error_message(op, left, right)

    op             is a method or a function
    left           is a geo object
    right          is probably a geo object.

    message        is what did not work
    """
    return "%s(%s, %s)"%(op.__name__,
                         left.__class__.__name__,
                         right.__class__.__name__)
