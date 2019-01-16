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



"""Ths module is full of base classes for coordinates. Without an ellipsoid,
they don't work. When you get an ellipsoid, e.g., wgs84, then use the factory
function:


ECEF, LLH, LTP, SCH =  CoordinateFactory(wgs84)

The 4 clases are:
earth centered earth fixed,
lat, lon, height,
local tangent plane
S, C, H

The later 2 require a PegPoint to define the origin and direction. The classes
are completely polymorphic:

If you have an instance p:

p.ecef()
p.llh()
p.ltp(peg_point=None)
p.sch(peg_point=None)

which gove the same point in a new coordinate system. Doesn't matter what p is
to start with. Likewise, of you need to change an ellipsoid, say, to airy1830,
just do:

p_new = p>>airy1830

Doesn't matter what p is, p_new will be the same type of coordinate (but a
different class, of course).

Note on Affine spaces: you can difference points and get vectors. You can add
vector to points. But you can't add two points.

Note: all coordinates have a cartesian counter part, which for a cartesian
system has the same orientation and origin:

LLH.cartesian_counter_part --> ECEF
SCH.cartesian_counter_part --> LTP   (with the same PegPoint)

With that, subtraction is defined for all coordinates classes:

v = p2 -p1

will give a vector in p2's cartesian counter part, and the inverse operation:

p2 = p1 +v

will give a new point p2 in the same type a p1. (So if p1 is in SCH, then v
will be interpreted as a vector in the congruent LTP). It's all about
polymorphism. If you don't like it, then write your code explcitly.



Methods:
--------
Coordinates have all the generalized methods described in euclid.__doc__, so
see that if you need to take an average, make an iterator, a list, or what not.

p1.bearing(p2)
p1.distance(p2)   compute bearing and distance bewteen to points on the
                  ellipsoid.

p.latlon()        get the latitude longtude tuple.
p.vector()        make a euclid.Vector
p.space_curve()   makes a motion.SpaceCurve


Transformation functions are computed on the fly, and are availble via:

ECEF.f_ecef2lla
ECEF.affine2ltp

LLH.f_lla2ecef

LTP.f_ltp2sch
LTP.affine2ecef

SCH.f_sch2ltp


these are either affine.Affine transformations, or nested dynamic functions--
eitherway, they do not exisit unit requested, and at that point, the get made,
called, and forgotten.
"""

## \namespace geo::coordinates Mapping coordinates in search of an
## geo.ellipsoid.Ellipsoid
from operator import methodcaller
from collections import namedtuple

import numpy as np

from isceobj.Util.geo import euclid
from isceobj.Util.geo import charts


## Function that converts a coordinate instance to ECEF
to_ecef = methodcaller('ecef')
## Function that converts a coordinate instance to LLH
to_llh = methodcaller('llh')
## Function that converts a coordinate instance to LTP
to_ltp = lambda inst, peg_point=None: inst.ltp(peg_point=peg_point)
## Function that converts a coordinate instance to SCH
to_sch = lambda inst, peg_point=None: inst.sch(peg_point=peg_point)


## <a href="http://mathworld.wolfram.com/AliasTransformation.html">Alias
## transformation</a> from
## <a href="http://en.wikipedia.org/wiki/North_east_down">North East Down</a>
## to <a href="http://en.wikipedia.org/wiki/Geodetic_system#Local_east.2C_north.2C_up_.28ENU.29_coordinates">East North Up</a>
ned2enu = charts.Roll(90).compose(charts.Pitch(90))

## <a href="http://mathworld.wolfram.com/AliasTransformation.html">Alias
## transformation</a> to <a href="http://en.wikipedia.org/wiki/North_east_down">
## NED</a> from <a href="http://en.wikipedia.org/wiki/Geodetic_system#Local_east.2C_north.2C_up_.28ENU.29_coordinates">ENU</a>
enu2ned = ~ned2enu

## A peg point is simple enough to be a more than a
##<a href="http://docs.python.org/library/collections.html#collections.namedtuple">
## namedtuple</a>.
PegPoint = namedtuple("PegPoint", "lat lon hdg")

## Latitude and Longitude pairs are common enough to get a namedtuple.
LatLon = namedtuple("LatLon", "lat lon")


## Rotate from ECEF's coordiantes to an East North Up LTP system\n The
## rotation depends on latitude and longitude, but is independent of the
## ellipsoid.Ellipsoid\n Computed from:\n\n charts.Pitch() by longitude
## followed by \n chars.Roll() minus latitude.
def rotate_from_ecef_to_enu(lat, lon):
    """rotate_from_ecef_to_enu(lat, lon)

    Parameters
    -----------

    lat :  array_like    latitude (degrees)
    lon :  array_like    longitude (degrees)

    Returns
    -------
    versor : array_like euclid.Versor representing the transformation from ECEF
    to ENU
    """
    ## Note the negative sign on the latitude
    return ned2enu.compose(charts.Pitch(lon)).compose(charts.Roll(-lat))

## Rotate from ECEF's coordiantes an LTP wit any heading\n (using
## rotate_from_ecef_to_enu() ) \n The rotation depends on latitude, longitude,
## and heading, but is independent of the ellipsoid.EllipsoidComputed from:\n\n
## rotate_from_ecef_to_enu()  of (lat, lon) followed by \n charts.Yaw() with
## yaw = -(heading-90)
def rotate_from_ecef_to_tangent_plane(lat, lon, hdg):
    """rotate_from_ecef_to_tangent_plane(lat, lon, hdg)

    Parameters
    -----------

    lat :  array_like    latitude (in degrees)
    lon :  array_like    longitude (in degrees)
    hdg:   array_like    heading (in degrees)

    Returns
    -------
    versor : array_like charts.Versor representing the transformation from
    ECEF to LTP
    """
    return rotate_from_ecef_to_enu(lat, lon).compose(charts.Yaw(-(hdg-90)))

## Bearing Between two Points specified by latitude and longitude \n
## \f$ b(\phi_1, \lambda_1, \phi_2, \lambda_2)= \tan^{-1}{\frac{\sin{(\lambda_2-\lambda_1)}\cos{\phi_2}}{(\cos{\phi_1}\sin{\phi_2}-\sin{\phi_1}\cos{\phi_2})\cos{(\phi_2-\phi_1)}}} \f$ \n
## (http://mathforum.org/library/drmath/view/55417.html)
def bearing(lat1, lon1, lat2, lon2):
    """hdg = bearing(lat1, lon1, lat2, lon2)

    lat1, lon1  are the latitude and longitude of the starting point
    lat2, lon2  are the latitude and longitude of the ending point

    hdg is the bearing (heading), in degrees, linking the start to the end.
    """
    from isceobj.Util.geo.trig import sind, cosd, arctand2
    dlat = (lat2-lat1)
    dlon = (lon2-lon1)
    y = sind(dlon)*cosd(lat2)
    x = cosd(lat1)*sind(lat2)-sind(lat1)*cosd(lat2)*cosd(dlon)

    return arctand2(y, x)

## The distance between 2 points, specified by latitude and longiutde--
## this depends on the ellipsoid, so you need to supply one, and you get a
## function back that compute the distance.
def get_distance_function(ellipsoid_):
    """distance = get_distance_function(ellsipoid_)(*args)  --that is:

    f = distance(ellipsoid_)

    ellipsoid_ is an ellipsoid.Ellipsoid instance
    f(*args)   is a callable function that return the ellipsoid.
    """
    return ellipsoid_.distance


## This must have factory function takes an ellipsoid and builds coordinate
## transformation classes around it\n Note: this really is a factory function:
## it makes new classes that heretofore DO NOT EXIST.
def CoordinateFactory(ellipsoid_, cls):
    """cls' = CoordinateFactory(ellipsoid_, cls)


    Takes an ellipsoid and creates new coordinate classes that have an
    ellipsoid attribute equal to the functions argument (ellipsoid_). Thus,
    the class's method will work, since they *need* and ellipsoid.

    Inputs:
    -------
    ellipsoid_      This should be an ellipsoid.Ellipsoid instance (see
                    ellipsoid.Ellipsoid.__init__ for call example)
    cls             a bare coorinate class

    Outputs:
    --------
    cls'            a coordinate class with an ellipsoid
    """
    class CyclicMixIn(object):
        ellipsoid=ellipsoid_
        pass

    return type(cls.__name__,
                (cls, CyclicMixIn,),
                {
            "__doc__":"""%s sub-class associated with ellipsoid model:%s""" %
            (cls.__name__, ellipsoid_.model)
            }
                )


## This for development of pickle compliant code
WARN = None

## "Private" base class for Coodinates with a fixed origin
class _Fixed(euclid.PolyMorphicNumpyMixIn):
    """_Fixed is a base class for coordinates with a fixed origin"""

    ## The default is alway "x" "y" "z"
    coordinates = ("x", "y", "z")

    ## The default is always meters
    UNITS = 3*("m",)

    ## These have no ::PegPoint, explicitly.
    peg_point = None

    ## Init 3 coordinates from class's cls.coordinates
    def __init__(self, coordinate1, coordinate2, coordinate3,
                 ellipsoid=None):
        for name, value in zip(
            self.__class__.coordinates,
            (coordinate1, coordinate2, coordinate3)
            ):
            setattr(self, name, value)
            pass
        self.ellipsoid = ellipsoid
        if self.ellipsoid is WARN: raise RuntimeError("no ellipsoid!")
        return None

    ## call super, but use "coordinates" instead of "__slots__" -- it didn't
    ## want coordinates to have __slots__
    def iter(self):
        return super(_Fixed, self).iter("coordinates")

    ## Make into a motion::SpaceCurve().
    def space_curve(self):
        return self.vector().space_curve()

    ## Subtraction ALWAYS does vector conversion in the left arguments
    ## cartesian frame, appologies for the if-then block
    def __sub__(self, other):
        """point1 - point2 gives the vector pointing from point1 to point2,
        in point1's cartesian coordinate system"""
        try:
            if self.peg_point == other.peg_point:
                return self.vector() - other.vector()
            else:
                if self.peg_point is None:
                    return self.vector() - other.ecef().vector()
                else:
                    return self.vector() - other.ltp(self.peg_point).vector()
                pass
            pass
        except AttributeError as err:
            if isinstance(other, euclid.Vector):
                from isceobj.Util.geo.exceptions import AffineSpaceError
                raise AffineSpaceError
            raise err
        pass


    ## You can only add a vector, in the coordaintes cartesian frame, and you
    ## get back the same coordiantes--
    def __add__(self, vector):
        """point1 + vector gives point2 in point1's coordinate system, with
        the vector interpreted in point1's cartesian frame"""
        result = self.cartesian_counter_part().vector() + vector
        if isinstance(self, _Cartesian):
            kwargs =  {"ellipsoid":self.ellipsoid}
            if self.peg_point:
                kwargs.update({"peg_point":self.peg_point})
                pass
            return self.__class__( *(result.iter()), **kwargs )
        else:
            if isinstance(self, _Pegged):
                return self.ellipsoid.LTP(
                    *(result.iter()), peg_point = self.peg_point
                     ).sch()
            else:
                return self.ellipsoid.ECEF( *(result.iter()) ).llh()
            pass
        pass


    ## Object is iterbale ONLY if its attributes are, Use with care
    def __getitem__(self, index):
        """[] --> index over components' iterator, it is NOT a tensor index"""

        return self.__class__(*[item[index] for item in self.iter()],
                               ellipsoid=self.ellipsoid)

    ## This allow you to send instance to the next function
    def next(self):
        return self.__class__(*map(next, self.iter()),
                               ellipsoid=self.ellipsoid)

    ## The iter() function: returns an instance that is an iterator
    def __iter__(self):
        return self.__class__(*map(iter, self.iter()),
                               ellipsoid=self.ellipsoid)

    ## string
    def __str__(self):
        result = ""
        for name, value in zip(self.__class__.coordinates, self.components()):
            result += name+":"+str(value)+"\n"
            pass
        return result

    ## repr() and str() are now the same.
    __repr__ = __str__

    ## crd>>ellspoid put coordinates on an ellipsoid
    def __rshift__(self, other):
        return self.change_ellipsoid(other)

    ## Get likenamed class on another ellipsoid.Ellipsoid by using
    ## methodcaller and the class's __name__\n This avoids eval or exec calls.
    def change_ellipsoid(self, other):
        # figure out the name of the method to convert to correct coordinates
        # and make a function that calls it- since all conversion methods are
        # lower case versions of the target class, this works:
        method = methodcaller(self.__class__.__name__.lower())
        # ECEF in this ellipsoid
        here_ecef = self.ecef()
        # ECEF if that ellipsoid -- they should be he same, numerically, of
        # course, they are DIFFERENT classes-- and that's why it works
        there_ecef = other.ECEF(*(here_ecef.iter()))

        # no peg point result, just call the method converting to the right
        #coordinates and boom, you're done.
        if (not hasattr(self, "peg_point")) or self.peg_point is None:
            return method(there_ecef)

        # NOTE: new_peg IS NOT the same point as peg_point: it can't be.
        new_peg = self.ellipsoid.LLH(self.peg_point.lat, self.peg_point.lon, 0.)
        new_peg = PegPoint(new_peg.lat, new_peg.lon, self.peg_point.hdg)

        # now call the method on the new ellipsoid instance, this time with a
        # peg_point kwarg.
        return method(there_ecef, peg_point=new_peg)

    ## geo.egm96.geoid call, currently forces ellipsoid to WGS84- not sure how
    ## to deal otherwise
    def egm96(self, force=True):
        """get egm96 heights at lat & lon -- you will
        be forced onto WGS84."""
        raise NotImplementedError
        from isceobj.Util.geo import egm96
        llh = self.llh()
        if force:
            from ellipsoid import WGS84
            llh>>WGS84
            pass
        return geoid(self.lat, self.lon)

    def __neg__(self):
        return tuple(self.__class__.__bases__[1].__neg__(self).tolist())

    pass

## "Private" base class for Coordinates with a variable origin (a ::PegPoint)
class _Pegged(_Fixed):
    """_Pegged is a base class for coordinates with a variable origin"""

    def __init__(self, coordinate1, coordinate2, coordinate3,
                 peg_point=None, ellipsoid=None):
        super(_Pegged, self).__init__(coordinate1,
                                      coordinate2,
                                      coordinate3,
                                      ellipsoid=ellipsoid)
        self.peg_point = peg_point
        return None

    def new(self, x, y, z):
        return self.__class__(x, y, z, peg_point=self.peg_point,
                              ellipsoid=ellipsoid)


    __todo__ = """Because of the pegoint, the sequence/iterator methods need a
                   pegpoint kwarg-- it should be simpler """

    ## Object is iterbale ONLY if its attributes are, Use with care
    def __getitem__(self, index):
        """[] --> index over components' iterator, it is NOT a tensor index"""

        return self.__class__(*[item[index] for item in self.iter()],
                               peg_point=self.peg_point,
                               ellipsoid=self.ellipsoid)

    ## This allow you to send instance to the next function
    def next(self):
        return self.__class__(*map(next, self.iter()),
                               peg_point=self.peg_point,
                               ellipsoid=self.ellipsoid)

    ## The iter() function: returns an instance that is an iterator
    def __iter__(self):
        return self.__class__(*map(iter, self.iter()),
                               peg_point=self.peg_point,
                               ellipsoid=self.ellipsoid)

    ## str: call super and added a peg point.
    def __str__(self):
        return super(_Pegged, self).__str__()+str(self.peg_point)

    ## Just a synonym
    @property
    def peg(self):
        return self.peg_point


    ## Call super and add the peg_point after the fact
    def broadcast(self, func, *args, **kwargs):
        result = super(_Pegged, self).broadcast(func, *args, **kwargs)
        result.peg_point = self.peg_point
        return result

    ## Add a PegPoint to _Fixed.change_ellipsoid
    def change_ellipsoid(self, other):
        result = super(_Pegged, self).change_ellipsoid(other)
        result.peg_point = self.peg_point
        return result

#    def __neg__(self):
#        return (super(_Peged, self),__neg__(), {'peg_point':self.peg_point})


    pass


## a coordinate base class.
class _C(object):

    def __neg__(self):
        return (-self.vector()).iter()

    ## Convert to LLH and get the bearing bewteen two coordinates, see module
    ## function bearing() for more.
    def bearing(self, other):
        """p1.bearing(p2) will compute the heading from p1 to p2 """
        return bearing(*(self.llh().latlon()+other.llh().latlon()))

    ## Get distance between nadir points on the Ellipsoid, self
    ## ellipsoid.Ellipsoid.distance()
    def distance_spherical(self, other):
        """calls self.ellipsoid.distance_spherical"""
        return self.ellipsoid.distance_spherical(*(
                self.llh().latlon()+other.llh().latlon()
                ))
    def distance_true(self, other):
        """calls self.ellipsoid.distance_true"""
        return self.ellipsoid.distance_true(*(
                self.llh().latlon()+other.llh().latlon()
                                             ))
    ## pick a distance algorithm
    distance = distance_true

    ## Make a named tuple
    def latlon(self):
        """makes a lit on namedtuple"""
        return LatLon(*(self.llh().tolist()[:-1]))

    pass

## A "private" mixin for cartesian coordinates
class _Cartesian(_C):
    """A mix-in for cartesian coordinates"""

    ## convert result to a Vector relative to origin
    def vector(self, peg_point=None):
        """vector([peg_point=None])

        will convert self.x, self.y, self.z into a euclid.Vector if peg_point
        is None, otherwise, it will transform to the LTP defined by the
        peg_point and the call vector(None).
        """
        if peg_point is None:
            return euclid.Vector(*self.iter())
        else:
            return self.ltp(peg_point=peg_point).vector(peg_point=None)
        pass

    ## This method is trival, and allow polymorphic calls with _NonCartesian
    ## instances -- should be coded as cartesian_counter_part =
    ## PolymorphicNumpyMixIn.__pos__, but that is TBD.
    def cartesian_counter_part(self):
        return self

    pass


## A "private" mixin for non cartesian coordinate systems
class _NonCartesian(_C):
    """A mixin for non cartesion classes, this brings in the "vector" method."""

    ## Convert to the class's Cartesian counter part's vector
    def vector(self, peg_point=None):
        """vector([peg_point=None]) will convert to the
        "cartesian_counter_part()" and then call vector(peg_point).

        See coordinates._Cartesian.vector.__doc__

        for more."""
        return self.cartesian_counter_part().vector(peg_point)

    pass


## A base class for <a href="http://en.wikipedia.org/wiki/ECEF">Earth Centered
## Earth Fixed</a> coordinates.
class ECEF(_Fixed, _Cartesian):
    """ECEF(x, y, z)

    Earth Centered Earth Fixed Coordinates.

    The x axis goes from the center to (lat=0, lon=0)
    The y axis goes from the center to (lat=0, lon=90)
    The z axis goes from the center to (lat=90)

    Methods to tranaform coordinates are:

    ecef()
    llh()
    ltp(peg_point)
    sch(peg_point)

    Other methods are:

    vector()            convert to a Vector object


    """

    ## Trival transformation
    ecef = _Cartesian.cartesian_counter_part

    ## ECEF --> LLH  via f_ecef2lla().
    def llh(self):
        """ecef.llh() puts cordinates in LLH"""
        return self.ellipsoid.LLH(*(self.f_ecef2lla(*self.iter())))

    ## ECEF --> LTP via affine2ltp() .
    def ltp(self, peg_point):
        """ecef.ltp(peg_point)  put coordinates in LTP at peg_point"""
        return self.ellipsoid.LTP(
            *(self.affine2ltp(peg_point)(self.vector()).iter()),
             peg_point=peg_point
             )

    ## ECEF --> LTP --> SCH   (derived)
    def sch(self, peg_point):
        """ecef.sch(peg_point)  put coordinates in SCH at peg_point"""
        return self.ltp(peg_point).sch()

    ## This readonly attribute is a function, f, that does:\n
    ## (lat, lon, hgt) = f(x, y, z) \n for a fixed Ellipsoid using
    ## ellipsoid.Ellipsoid.XYZ2LatLonHgt .
    @property
    def f_ecef2lla(self):
        return self.ellipsoid.XYZ2LatLonHgt

    ## This method returns the euclid::affine::Affine transformation from the
    ## ECEF frame to LTP at a ::PegPoint using
    ## ellipsoid.Ellipsoid.affine_from_ecef_to_tangent
    def affine2ltp(self, peg):
        try:
            result =  self.ellipsoid.affine_from_ecef_to_tangent(peg.lat,
                                                                 peg.lon,
                                                                 peg.hdg)
        except AttributeError as err:
            if peg is None:
                from isceobj.Util.geo.exceptions import AffineSpaceError
                msg = """Attempt a coordinate conversion to an affine space
                      with NO ORIGIN: peg_point is None"""
                raise AffineSpaceError(msg)
            raise err
        return result

    def __neg__(self):
        return ECEF(*super(ECEF, self).__neg__())

    pass

##  A base class for
## <a href="http://en.wikipedia.org/wiki/Geodetic_coordinates#Coordinates">
## Geodetic Coordinates</a>: <a href="http://en.wikipedia.org/wiki/Latitude">
## Latitue</a>, <a href="http://en.wikipedia.org/wiki/Longitude">Longitude</a>,
## <a href="http://en.wikipedia.org/wiki/Elevation">Height</a>.
class LLH(_Fixed, _NonCartesian):
    """LLH(lat, lon, hgt):

    Geodetic Coordinates: geodetic latitude.

    lat --> latitude in degrees (NEVER RADIANDS, EVER)
    lon --> longitude in degrees (NEVER RADIANDS, EVER)
    hgt --> eleveation, height, hgtitude in meters


    Methods are:

    ecef()
    llh()
    ltp(peg_point)
    sch(peg_point)
    """
    ## Geodetic coordinate names
    coordinates = ("lat", "lon", "hgt")

    ## Units are as is
    UNITS = ("deg", "deg", "m")

    ## LLH --> ECEF via f_lla2ecef()
    def ecef(self):
        """llh.ecef()  put coordinates in ECEF"""
        return self.ellipsoid.ECEF(*self.f_lla2ecef(*(self.iter())))

    ## LLH's counter part is ECEF
    cartesian_counter_part = ecef

    ## Trivial
    llh = euclid.PolyMorphicNumpyMixIn.__pos__

    ## LLH --> ECEF --> LTP
    def ltp(self, peg_point):
        """llh.ltp(peg_point)  put coordinates in LTP at peg_point"""
        return self.ecef().ltp(peg_point)

    ## LLH --> ECEF --> LTP --> SCH
    def sch(self, peg_point):
        """llh.sch(peg_point)  put coordinates in SCH at peg_point"""
        return self.ecef().sch(peg_point)

    ## This readonly attribute is a function, f,  that does:\n (x, y, z) =
    ## f(lat, lon, hgt) \n using ellipsoid.Ellipsoid.LatLonHgt2XYZ
    @property
    def f_lla2ecef(self):
        return self.ellipsoid.LatLonHgt2XYZ

    ## under construction
    def __neg__(self):
        return LLH(*super(LLH, self).__neg__())

    ## h is hgt
    @property
    def h(self):
        return self.hgt

    ## hardocde conversion
    @staticmethod
    def radians(x):
        """degs--> radians"""
        return 0.017453292519943295*x

    ## Convert to a local peg point (Default for NEU)
    def to_peg(self, hdg=90.):
        """peg_point = llh.to_peg([hdg=90.])

        peg_point = PegPoint(llh.lat, llh.lon, hdg)
        """
        return PegPoint(self.lat, self.lon, hdg)



    ## \f$ {\bf n}^e = \left[ \begin{array}{c} \cos{\phi}\cos{\lambda} \\ \cos{\phi}\sin{\lambda} \\ \sin{\phi}  \end{array} \right] \f$ \n
    ##  is the <a href="http://en.wikipedia.org/wiki/N-vector>N-vector</a>.
    def n_vector(self):
        """Compute a Vector instance representing the N-Vector"""
        from isceobj.Util.geo.trig import sind, cosd
        return euclid.Vector(
            cosd(self.lat)*cosd(self.lon),
            cosd(self.lat)*sind(self.lon),
            sind(self.lat)
            )

    pass

##  A base class for Local Tangent Plane Cartesian coordinates
class LTP(_Pegged, _Cartesian):
    """LTP(x, y, z, peg_point=<peg_point>)

    A point or points in a cartesian versions of SCH coordinates

    ARGS:
    ____
    x   along track cartesian coordinate in meters
    y   cross track cartesian coordinate in meters
    z   height above the ellipsoid in meters, at origin.


    KWARGS:
    ______
    peg_point      A PegPoint instance defining the coordinate system.


    Methods are:

    ecef()
    llh()
    ltp(peg_point=None)
    sch(peg_point=None)
    """

    ## LTP --> ECEF via ellipsoid.Ellipsoid.affine_from_tangent_to_ecef .
    def ecef(self):
        """ltp.ecef() put it into ECEF"""
        return self.ellipsoid.ECEF(*(self.affine2ecef()(self.vector()).iter()))

    ## LTP --> ECEF --> LLH
    def llh(self):
        """ltp.llh() put it into LLH"""
        return self.ecef().llh()

    ## Trivial OR LTP --> ECEF --> LTP'
    def ltp(self, peg_point=None):
        """ltp.ltp(peg_point) transforms to a new peg_point"""
        return self if peg_point is None else self.ecef().ltp(peg_point)

    ## LTP --> SCH via ellipsoid.Ellipsoid.TangentPlane2TangentSphere OR
    ## LTP --> LTP' --> SCH
    def sch(self, peg_point=None):
        """ltp.ltp(peg_point=None) transforms to SCH, possibly in a different
        peg"""
        return (
            self.ellipsoid.SCH(*self.to_sch_tuple, peg_point=self.peg_point)
            if peg_point is None else
            self.ltp(peg_point).sch(None)
            )

    ## This readonly attribute is a function, f,  that does:\n
    ## (s, c, h) = f(x, y, z) \n for this coordinate's ::PegPoint
    @property
    def f_ltp2sch(self):
        return self.ellipsoid.TangentPlane2TangentSphere(*self.peg_point)

    ## This readonly attribute computes a tuple of (s, c, h), computed from
    ## f_ltp2sch() .
    @property
    def to_sch_tuple(self):
        return self.f_ltp2sch(*self.iter())

    ## return the euclid::affine::Affine transformation to ECEF coordinates
    def affine2ecef(self):
        return self.ellipsoid.affine_from_tangent_to_ecef(*self.peg_point)

    pass


##  A base class for Local Tangent Sphere coordinates
class SCH(_Pegged, _NonCartesian):
    """SCH(s, c, h, peg_point=<peg_point>)

    A point or points in the SCH coordinate system:

    ARGS:
    ____
    s   along track polar coordinate in meters
    c   cross track polar coordinate in meters
    h   height above the ellipsoid in meters (at origin).


    KWARGS:
    ______
    peg_point      A PegPoint instance defining the coordinate system.



    Methods are:

    ecef()
    llh()
    ltp(peg_point=None)
    sch(peg_point=None)
    """
    ## These non-cartesian coordinates are called S, C and H.
    coordinates = ("s", "c", "h")

    ## SCH --> LTP --> ECEF
    def ecef(self):
        return self.ltp(None).ecef()

    ## SCH --> LTP --> LLH
    def llh(self):
        return self.ltp(None).llh()

    ## SCH --> LTP using to_ltp_tuple() if peg_point is None, oherwise:
    ## SCH --> ECEF --> LTP'
    def ltp(self, peg_point=None):
        return (
            self.ellipsoid.LTP(*self.to_ltp_tuple, peg_point=self.peg_point)
            if peg_point is None else
            self.ecef().ltp(peg_point)
            )

    ## SCH as a vector goes to the LTP
    cartesian_counter_part = ltp

    ## Trivial if peg_point is None, otherwise: SCH --> ECEF --> SCH'
    def sch(self, peg_point=None):
        return self if peg_point is None else self.ecef().sch(peg_point)

    ## This readonly attribute is  a function, f, that does:\n (x, y, z) =
    ## f(s, c, h) \n for this coordinate's ::PegPoint, using
    ## ellipsoid.Ellipsoid.TangentSphere2TangentPlane.
    @property
    def f_sch2ltp(self):
        return self.ellipsoid.TangentSphere2TangentPlane(*self.peg_point)

    ## This readonly attribute is  a tuple of (x, y, z) computed from
    ## f_sch2ltp().
    @property
    def to_ltp_tuple(self):
        return self.f_sch2ltp(*self.iter())

    pass

## A Tuple of supported coordinates
FRAMES = (ECEF, LLH, LTP, SCH)
