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



"""Ellipsoid do a lot. The hold cordinates, transformation functions,
affine computations, and distance computations, latitude converstions,

"""
## \namespace geo::ellipsoid Ellipsoid's and their geo::coordinates
__date__ = "10/30/2012"
__version__ = "1.2"


from isceobj.Util.geo import coordinates
from isceobj.Util.geo.trig import cosd, sind, arctand2, arctand, tand
np = coordinates.np
arctan = np.arctan
arctan2 = np.arctan2
arccos = np.arccos
arcsin = np.arcsin


from isceobj.Util.geo import euclid
from isceobj.Util.geo import charts
Vector = euclid.Vector
Matrix = euclid.Matrix
from isceobj.Util.geo.affine import Affine
Roll = charts.Roll
Pitch = charts.Pitch
Yaw = charts.Yaw
Polar = euclid.Polar


## \f$ b = a\sqrt{1-\epsilon^2} \f$
def a_e2_to_b(a, e2):
    return a*(1.-e2)**0.5

## \f$ f = 1-\frac{b}{a} \f$
def a_e2_to_f(a, e2):
    return (a-a_e2_to_b(a, e2))/a

## \f$ f = f^{-1} \f$
def a_e2_to_finv(a, e2):
    return 1/a_e2_to_f(a, e2)


## A 2-parameter Oblate Ellipsoid of Revolution
class _OblateEllipsoid(object):
    """This class is a 2 parameter oblate ellipsoid of revolution and serves
    2 purposes:

    __init__  defines the shape, see it for signature

    all the other methods compute the myraid of "other" ellipsoid paramters
    in the literature, and they provide conversion from the common latitude
    to all the other latitudes out there.
    """

    ## This __init__ defines the shape
    def __init__(self, a=1.0, e2=0.0, model="Unknown"):
        """(a=1, e2=0, model="Unknown")

        a  is the semi major axis
        e2 is the square of the 1st eccentricity: (a**2-b**2)/a**@
        """
        ## The semi-major axes
        self._a = a
        ## The first eccentricty squared.
        self._e2 = e2
        ## The model name
        self._model = model
        return None

    ## Test equality of parameters-- I use _OblateEllipsoid.a and
    ## _OblateEllipsoid.b for simplicity
    def __eq__(self, other):
        return (self.a == other.a) and (self.b == other.b)

    ## Test inequality of parameters-- I use Ellipsoid.a and Ellipsoid.b for
    ## simplicity
    def __ne__(self, other):
        return (self.a != other.a) or (self.b != other.b)

    ## Semi Major Axis
    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        try:
            if float(value)<=0:
                raise ValueError(
                    "Semi-major axis must be positive and finite, not:%s" %
                    str(value)
                    )
        except (TypeError, AttributeError):
            raise TypeError(
                "Semi-major axis must be a float, or have a __float__ method"
                )
        self._a = float(value)
        pass

    ## 1st eccentricity squared
    @property
    def e2(self):
        return self._e2

    @e2.setter
    def e2(self, value):
        try:
            if not (0 <= value <1):
                raise ValueError(
                    "First Eccentricity Squared must be on [0, 1)" %
                    str(value)
                    )
        except (TypeError, AttributeError):
            raise TypeError(
                "Semi-major axis must be a float, or have a __float__ method"
                )
        self._e2 = value
        pass

    ## \f$ \epsilon = \sqrt{1-{\frac{b}{a}}^2}  \f$\n First Eccentricity
    @property
    def e(self):
        return self.e2**0.5

    @e.setter
    def e(self, value):
        self.e2 = value**2
        pass

    @property
    def b(self):
        return a_e2_to_b(self.a, self.e2)

    @b.setter
    def b(self, value):
        self.e = 1.-(value/self.a)**2
        pass

    @property
    def finv(self):
        return a_e2_to_finv(self.a, self.e2)

    @finv.setter
    def finv(self, value):
        self.f = 1/value
        pass

    ## \f$ f=1-\cos{oe} \f$ \n Flatenning
    @property
    def f(self):
        return a_e2_to_f(self.a, self.e2)

    @f.setter
    def f(self, value):
        self.e = 1. - self.a*value
        pass

    ## \f$\cos{oe} = b/a \f$ \n Cosine of the
    ## <a href="http://en.wikipedia.org/wiki/Angular_eccentricity">
    ## angular eccentricity</a>.
    @property
    def cosOE(self):
        return self.b/self.a

    @cosOE.setter
    def cosOE(self, value):
        self.b = value*self.a
        pass

    ## \f$ f' \equiv n =   \tan^2{\frac{oe}{2}} = \frac{a-b}{a+b} \f$\n
    ## <a href=" http://en.wikipedia.org/wiki/Flattening">The Second
    ## Flattening</a>.
    @property
    def f_prime(self):
        return self.f/(1.+self.cosOE)

    ## \f$ n = \frac{a-b}{a+b} \f$ \n Third Flattening
    @property
    def f_double_prime(self):
        return (self.a-self.b)/(self.a+self.b)

    ## \f$n \equiv f'' \f$
    n = f_double_prime

    ## \f$ \epsilon' = \sqrt{{\frac{a}{b}}^2-1}  \f$\n Second Eccentricity
    @property
    def e_prime(self):
        return self.e_prime_squared**0.5

    ## \f$ \epsilon'^2 \f$\n Second Eccentricity Squared
    @property
    def e_prime_squared(self):
        return (self.cosOE**2-1.)

    ## \f$ e'' = \sqrt{ \frac{a^2-b^2}{a^2+b^2} } \f$ \n Third Eccentricity
    @property
    def e_double_prime(self):
        return self.e_double_prime_squared**0.5

    ## \f$ e''^2 =  \frac{a^2-b^2}{a^2+b^2}  \f$ \n Third Eccentricity Squared
    @property
    def e_double_prime_squared(self):
        return (self.a**2-self.b**2)/(self.a**2+self.b**2)

    ##  \f$ R_1 \f$, \n
    ## <a href="http://en.wikipedia.org/wiki/Earth_radius#Mean_radius:_R1">
    ## Mean Radius</a>.
    @property
    def R1(self):
        return (2.*self.a+self.b)/3.

    ##  \f$ R_2 \f$, \n
    ## <a href="http://en.wikipedia.org/wiki/Earth_radius#Authalic_radius:_R2">
    ## Authalic Radius</a>
    @property
    def R2(self):
        return NotImplemented

    ## \f$ R_3 \f$ \n
    ##<a href="http://en.wikipedia.org/wiki/Earth_radius#Volumetric_radius:_R3">
    ## Volumetric Radius</a>
    @property
    def R3(self):
        return (self.b*self.a**2)**(1./3.)


    ## \f$\eta' = 1/\sqrt{1-\epsilon^2\sin^2{\phi}} \f$;\n
    ## <a href="http://en.wikipedia.org/wiki/Latitude#Elliptic_parameters">
    ## Inverse of the principle elliptic integrand</a>
    def eta_prime(self, lat):
        """Inverse of the principle elliptic integrand (lat/deg)"""
        return NotImplemented

    ## \f$ \frac{\pi}{180^{\circ}} M(\phi) \f$ \n
    ## <a href="http://en.wikipedia.org/wiki/Latitude#Degree_length">
    ## Latitude degree length</a>
    def latitude_degree_length(self, lat):
        """Length of a degree of latitude (deg-->m) """
        return np.radians(self.meridional_radius_of_curvature(lat))

    ## \f$ \frac{\pi}{180^{\circ}} \cos{(\phi)} N(\phi) \f$\n
    ## <a href="http://en.wikipedia.org/wiki/Latitude#Degree_length">
    ## Longitude degree length</a>
    def longitude_degree_length(self, lat):
        """Length of a degree of longitude (deg-->m)"""
        from isceobj.Util.geo.trig import cosd
        return np.radians(cosd(lat) * self.normal_radius_of_curvature(lat))

    ##\f$ M=M(\phi)=\frac{(ab)^2}{[(a\cos{\phi})^2+(b\sin{\phi})^2 ]^{\frac{3}{2}} }\f$
    def meridional_radius_of_curvature(self, lat):
        """North Radius (northRad): Meridional radius of curvature (M),
        meters for latitude in degress """
        return (
            (self.a*self.b)**2/
            ( (self.a*cosd(lat))**2 + (self.b*sind(lat))**2 )**1.5
            )


    ## \f$N(\phi) = a \eta' \f$, \n
    ## <a href="http://en.wikipedia.org/wiki/Earth_radius#Normal">Normal Radius
    ## of Curvature</a>
    def normal_radius_of_curvature(self, lat):
        """East Radius (eastRad): Normal radius of curvature (N), meters for
        latitude in degrees """
        return (
            self.a**2/
            ( (self.a*cosd(lat))**2 + (self.b*sind(lat))**2 )**0.5
            )

    ## Synonym for ::normal_radius_curvature
    eastRad = normal_radius_of_curvature

    ##\f$ \frac{1}{R(\phi,\alpha)}  = \frac{\cos^2{\alpha}}{M(\phi)} + \frac{\sin^2{\alpha}}{N(\phi)} \f$ \n
    ## Radius of curvature along a bearing.
    def local_radius_of_curvature(self, lat, hdg):
        """local_radius_of_curvature(lat, hdg)"""
        return 1./(
            cosd(hdg)**2/self.M(lat) +
            sind(hdg)**2/self.N(lat)
            )

    localRad = local_radius_of_curvature

    ## \f$ N(\phi) \f$ \n Normal Radius of Curvature
    N = normal_radius_of_curvature

    ## \f$ M(\phi) \f$ \n Meridional Radius of Curvature
    M = meridional_radius_of_curvature

    ## Synonym for ::meridional_radius_curvature
    northRad = M

    ## \f$ R=R(\phi)=\sqrt{\frac{(a^2\cos{\phi})^2+(b^2\sin{\phi})^2}{(a\cos{\phi})^2+(b\sin{\phi})^2}}\f$\n
    ## Radius at a given geodetic latitude.
    def R(self, lat):
        return (
            ((self.a**2*cosd(lat))**2 + (self.b**2*sind(lat))**2)/
            ((self.a*cosd(lat))**2 + (self.b*sind(lat))**2)
            )**0.5

    ## \f$ m(\phi) = a(1-e^2)\int_0^{\phi}{(1-e^2\sin^2{x})^{-\frac{3}{2}}dx} \f$
    def m(self, phi):
        try:
            from scipy import special
            f = special.ellipeinc
        except ImportError as err:
            f = NotImplemented # you can add you ellipeinc here, and the code will work.
            msg = "This non-essential method requires scipy.special.ellipeinc"
            raise err(msg)
        return (
            f(phi, self.e2) -
            self.e2*np.sin(phi)*np.cos(phi)/np.sqrt(1-self.e2*np.sin(phi)**2)
            )


    ## \f$ \chi(\phi)=2\tan^{-1}\left[ \left(\frac{1+\sin\phi}{1-\sin\phi}\right) \left(\frac{1-e\sin\phi}{1+e\sin\phi}\right)^{\!\textit{e}} \;\right]^{1/2} -\frac{\pi}{2} \f$

    ## \n <a href="http://en.wikipedia.org/wiki/Latitude#Conformal_latitude">Conformal latitude</a>
    def common2conformal(self, lat):
        """Convert common latiude (deg) to conformal latiude (deg) """
        sinoe = np.sqrt(self.e2)
        sinphi = sind(lat)
        return (
            2.*arctand(
                np.sqrt(
                    (1.+sinphi)/(1.-sinphi)*(
                        (1.-sinphi*sinoe)/(1.+sinphi*sinoe)
                        )**sinoe
                    )
                ) - 90.0
            )

    ## \f$ \beta(\phi) = \tan^{-1}{\sqrt{1-e^2}\tan{\phi}} \f$ \n
    ## <a href="http://en.wikipedia.org/wiki/Latitude#Reduced_latitude">
    ## Reduced Latitude</a>
    def common2reduced(self, lat):
        """Convert common latiude (deg) to reduced latiude (deg) """
        return arctand( self.cosOE * tand(lat) )

    common2parametric = common2reduced

    ##\f$q(\phi)=\frac{(1-e^2)\sin{\phi}}{1-e^2\sin^2{\phi}}=\frac{1-e^2}{2e}\log{\frac{1-e\sin{\phi}}{1+e\sin{\phi}}}\f$
    def q(self, phi):
        """q(phi)"""
        sinphi = np.sin(phi)
        return (
            (1-self.e2)*sinphi/(1-self.e2*sinphi**2) -
            ((1-self.e2)/2/self.e)*np.log((1-self.e*sinphi)/(1+self.e*sinphi))
            )

    ## Latitude in radians
    @staticmethod
    def phi(lat):
        """function to convert degrees to radians"""
        return np.radians(lat)

    ## \f$ \xi = \sin^{-1}{\frac{q(\phi)}{q(\pi/2)}} \f$ \n
    ## <a href="http://en.wikipedia.org/wiki/Latitude#Authalic_latitude)">
    ## Authalic Latitude</a>
    def common2authalic(self, lat):
        """Convert common latiude (deg) to authalic latiude (deg) """
        phi_ = self.phi(lat)
        return np.degrees(np.arcsin(self.q(phi_)/self.q(np.pi/2.)))

    ## \f$ \psi(\phi) = \tan^{-1}{(1-e^2)\tan{\phi}} \f$
    ## \n <a href="http://en.wikipedia.org/wiki/Latitude#Geocentric_latitude">
    ## Geocentric Latitude</a>.
    def common2geocentric(self, lat):
        """Convert common latiude (deg) to geocentric latiude (deg) """
        return arctand( tand(lat) * self.cosOE**2 )

    ##  \f$ \mu{\phi} = \frac{\pi}{2}\frac{m(\phi)}{m(\phi/2)} \f$\n
    ## <a href="http://en.wikipedia.org/wiki/Latitude#Rectifying_latitude">
    ## rectifying latitude</a>
    def common2rectifying(self, lat):
        """Convert common latitude (deg) to rectifying latitude (deg) """
        return 90.*self.m(np.radians(lat))/self.m(np.radians(90))

    ## \f$\psi(\phi)=\sinh^{-1}{(\tan{\phi})-e\tanh^{-1}{(e\sin{\phi})}}\f$ \n
    ## <a href=" http://en.wikipedia.org/wiki/Latitude#Isometric_latitude">
    ## isometric latitude </a>
    def common2isometric(self, lat):
        """Convert common latitude (deg) to isometric latitude (deg) """
        phi_ = self.phi(lat)
        sinphi = np.sin(phi_)
        return np.degrees(
            np.log(np.tan(np.pi/4.+phi_/2.)) +
            (self.e/2.)*np.log( (1-self.e*sinphi)/(1+self.e*sinphi))
            )

    ## Geodetic latitude is the latitude.
    @staticmethod
    def common2geodetic(x):
        """x = common2geodetic(x)  (two names for one quantity)"""
        return x

    ## The bearing function is from coordiantes
    @staticmethod
    def bearing(lat1, lon1, lat2, lon2):
        """hdg = bearing(lat1, lon1, lat2, lon2)

        see coordinates.bearing
        """
        return coordinates.bearing(lat1, lon1, lat2, lon2)

    ## get c vs hdg, fit it and find correct zero-- make a pegpoint.
    def sch_from_to(self, lat1, lon1, lat2, lon2):
        """TBD"""
        hdg, c, h  = self._make_hdg_c(lat1, lon1, lat2, lon2)
        psi1 = float(np.poly1d(np.polyfit(hdg, c, 1)).roots)
        psi2 = np.poly1d(np.polyfit(hdg, c, 4)).roots

        iarg = np.argmin(abs(psi2-float(psi1)))

        psi = psi2[iarg]

        return coordinates.PegPoint(lat1, lon1, float(psi))


    ## get cross track coordinate vs. heading for +/-n degrees around
    ## spherical bearing
    def _make_hdg_c(self, lat1, lon1, lat2, lon2, n=1.):
        """also TBD"""
        c = []
        h = []

        p1 = self.LLH(lat1, lon1, 0.)
        p2 = self.LLH(lat2, lon2, 0.)

        b  = self.bearing(lat1, lon1, lat2, lon2)

        hdg = np.arange(b-n,b+n,0.001)
        for theta in hdg:
            peg = coordinates.PegPoint(lat1, lon1, float(theta))
            p = p2.sch(peg)
            c.append(p.c)
            h.append(p.h)
            pass
        c, h = map(np.array, (c,h))
        return hdg, c, h

    def distance_spherical(self, lat1, lon1, lat2, lon2):
        """d = distance(lat1, lon1, lat2, lon2)"""
        llh1 = self.LLH(lat1, lon1, 0.*lat1)
        llh2 = self.LLH(lat2, lon2, 0.*lat2)

        n1 = llh1.n_vector()
        n2 = llh2.n_vector()

        delta_sigma = arctan2( (abs(n1^n2)).w,  (n1*n2).w )

        return delta_sigma*self.R1

    def distance_sch(self, lat1, lon1, lat2, lon2):
        hdg = self.bearing(lat1, lon1, lat2, lon2)
        peg = coordinates.PegPoint(lat1, lon1, hdg)
        p2 = self.LLH(lat2, lon2, 0.).sch(peg)
        return p2

    ## Starting with: \n \f$ \lambda^{(n)} = \lambda_2-\lambda_1 \f$ \n
    ## iterate: \n
    ## \f$ \sin{\sigma} = \sqrt{(\cos{\beta_2}\sin{\lambda})^2+(\cos{\beta_1}\sin{\beta_2}-\sin{\beta_1}\cos{\beta_2}\cos{\lambda})^2} \f$ \n
    # \f$ \cos{\sigma} = \sin{\beta_1}\sin{\beta_2}+\cos{\beta_1}\cos{\beta_2}\cos{\lambda} \f$ \n
    ## \f$ \sin{\alpha} = \frac{\cos{\beta_1}\cos{\beta_2}\sin{\lambda}}{\sin{\sigma}} \f$ \n
    ## \f$ \cos{2\sigma_m} = \cos{\sigma}-\frac{2\sin{\beta_1}\sin{\beta_2}}{\cos^2{\alpha}} \f$ \n
    ## \f$ C = \frac{f}{16}\cos^2{\alpha}[4+f(4-3\cos^2{\alpha})] \f$ \n
    ## \f$ \lambda^{(n+1)} = L + (1-C)f\sin{\alpha}(\sigma+C\sin{\sigma}[\cos{2\sigma_m}+C\cos{\sigma}(-1+2\cos^2{2\sigma_m})]) \f$ \n \n
    ## Then, with: \n
    ## \f$ u^2 = \cos^2{\alpha} \frac{a^2-b^2}{b^2} \f$ \n
    ## \f$ A = 1 + \frac{u^2}{16384}(4096+u^2[-768+u^2(320-175u^2)]) \f$ \n
    ## \f$ B = \frac{u^2}{1024}(256-u^2[-128+u^2(74-47u^2)]) \f$ \n
    ## \f$ s = bA(\sigma-\Delta\sigma) \f$ \n
    ## \f$ \Delta\sigma = B\cdot\sin{\sigma}\cdot\big[\cos{2\sigma_m}+\frac{1}{4}B[\cos{\sigma}(-1+2\cos^2{2\sigma_m})-\frac{1}{6}B\cos{2\sigma_m}(-3+4\sin^2{\sigma})(-3+4\cos^2{2\sigma_m})]\big] \f$ \n
    ## see <a href="http://www.movable-type.co.uk/scripts/latlong-vincenty.html" target=_blank>Vincenty Formula</a>
    def great_circle(self, lat1, lon1, lat2, lon2):
        """s, alpha1, alpha2 = great_circle(lat1, lon1, lat2, lon2)

        (lat1, lon1)    p1's location
        (lat2, lon2)    p2's location

        s               distance along great circle
        alpha1          heading at p1
        alpha2          heading at p2
        """
        phi1, L1, phi2, L2 = lat1, lon1, lat2,lon2

        a = self.a
        f = self.f
        b = (1-f)*a
        U1 = self.common2reduced(phi1)  # aka beta1
        U2 = self.common2reduced(phi2)
        L = L2-L1

        lam = L

        delta_lam = 100000.

        while abs(delta_lam) > 1.e-10:
            sin_sigma = (
                (cosd(U2)*sind(lam))**2 +
                (cosd(U1)*sind(U2) - sind(U1)*cosd(U2)*cosd(lam))**2
                )**0.5
            cos_sigma = sind(U1)*sind(U2) + cosd(U1)*cosd(U2)*cosd(lam)
            sigma = arctan2(sin_sigma, cos_sigma)

            sin_alpha = cosd(U1)*cosd(U2)*sind(lam)/sin_sigma
            cos2_alpha = 1-sin_alpha**2

            cos_2sigma_m = cos_sigma - 2*sind(U1)*sind(U2)/cos2_alpha

            C = (f/16.)* cos2_alpha*(4.+f*(4-3*cos2_alpha))

            lam_new = (
                np.radians(L) +
                (1-C)*f*sin_alpha*(
                    sigma+
                    C*sin_sigma*(
                        cos_2sigma_m +
                        C*cos_sigma*(
                            -1+2*cos_2sigma_m**2
                             )
                        )
                    )
                )

            lam_new *= 180/np.pi

            delta_lam = lam_new-lam
            lam = lam_new
            pass

        u2 = cos2_alpha *(a**2-b**2)/b**2

        A_ = 1 + u2/16384*(4096+u2*(-768+u2*(320-175*u2)))
        B_ = u2/1024*(256+u2*(-128+u2*(74-47*u2)))

        delta_sigma = B_*sin_sigma*(
            cos_2sigma_m - (1/4.)*B_*(cos_sigma*(-1+2*cos_2sigma_m**2))-
            (1/6.)*B_*cos_2sigma_m*(-3+4*sin_sigma**2)*(-3+4*cos_2sigma_m**2)
            )

        s = b*A_*(sigma-delta_sigma)

        alpha_1 = 180*arctan2(
            cosd(U2)*sind(lam),
            cosd(U1)*sind(U2)-sind(U1)*cosd(U2)*cosd(lam)
            )/np.pi


        alpha_2 = 180*arctan2(
            cosd(U1)*sind(lam),
            -sind(U1)*cosd(U2)+cosd(U1)*sind(U2)*cosd(lam)
            )/np.pi

        return s, alpha_1, alpha_2

    ## Use great_circle() to get distance
    def distance_true(self, lat1, lon1, lat2, lon2):
        """see great_distance.__doc__"""
        return self.great_circle(lat1, lon1, lat2, lon2)[0]

    ## Use great_circle() to get initial and final bearings.
    def bearings(self, lat1, lon1,lat2, lon2):
        """see great_distance.__doc__"""
        return self.great_circle(lat1, lon1, lat2, lon2)[1:]

    ## Decide which one to use.
    distance = distance_true
    pass

## Just a place to put coodinate transforms
class EllipsoidTransformations(object):
    """This mixin is a temporary place to put transformations"""

    ## \f$ x = (N+h)\cos{\phi}\cos{\lambda} \f$ \n
    ## \f$ y = (N+h)\cos{\phi}\sin{\lambda} \f$ \n
    ## \f$ z= ((1-\epsilon^2)N+h) \sin{\phi} \f$ \n
    ## An analytic geodetic coordinates.LLH ->
    ## coordinates.ECEF calculation with tuple I/O, \n
    ## using method _OblateEllipsoid.N().
    def LatLonHgt2XYZ(self, lat, lon, h):
        """LatLonHgt2XYZ(lat, lon, h) --> (x, y, z)

        lat       is the latitude (deg)
        lon       is the longitude (deg)
        h         is the heigh (m)

        (x, y, z) is a tuple of ECEF coordinates (m)
        """
        N = self.N(lat)
        cos_lat = cosd(lat)
        return (
            cos_lat * cosd(lon) * (N+h),
            cos_lat * sind(lon) * (N+h),
            sind(lat) * ((1-self.e2)*N + h)
            )

    ## An iterative coordinates.ECEF -> coordinates.LLH calculation with tuple
    ## I/O, using ecef2llh().
    def XYZ2LatLonHgt(self, x, y, z, iters=10):
        """XYZ2LatLonHgt(x, y, z {iters=10})--> (lat, lon, hgt)

        calls module function:

        ecef2llh(self, x,y,  [, **kwargs])
        """
        return ecef2llh(self, x,y, z, iters=iters)


    ## Get the ::Vector from Earf's center to a latitude and longitude on the
    ## Ellipsoid
    def center_to_latlon(self, lat, lon=None, dummy=None):
        """center_to_latlon(lat {lon})

        Input:

        lat              is a PegPoint
        lat, lon         are latitude and longitude (degrees)


        Output:

        Vector           instance points from Core() to (lat, lon)
        """
        lat, lon, dummy = self._parse_peg(lat, lon, dummy)
        return self.LLH(lat, lon, 0.).ecef().vector()

    ## Compute ::euclid::affine::Affine() transform from coordinates.ECEF to
    ## coordinates.LTP (Tangent Plane)\n Work is done by
    ## coordinates.rotate_from_ecef_to_tangent_plane and center_to_latlon().
    def affine_from_ecef_to_tangent(self, lat, lon=None, hdg=None):
        """affine_from_ecef_to_tangent(lat {lon, hdg})

        Input:

        lat                   is a PegPoint
        lat, lon, hdg         are latitude, longitude, heading (degrees)


        Output:

        Affine  transform from Core to pegged tangent plane.
        """
        lat, lon, hdg = self._parse_peg(lat, lon, hdg)
        R = coordinates.rotate_from_ecef_to_tangent_plane(lat, lon, hdg)
        T = self.center_to_latlon(lat, lon)
        return Affine(R, -R(T))

    ## Compute ::Affine transform to ECEF from pegged'd LTP
    def affine_from_tangent_to_ecef(self, lat, lon=None, hdg=None):
        """affine_from_tangent_to_ecef(lat {lon, hdg})

        Input:

        lat                   is a PegPoint
        lat, lon, hdg         are latitude, longitude, heading (degrees)

        Output:

        Affine  transform from pegged tangent plane to Core in ECEF.
        """
        lat, lon, hdg = self._parse_peg(lat, lon, hdg)
        return ~(self.affine_from_ecef_to_tangent(lat, lon, hdg))

    ## A curried function that fixes peg point and returns a function from
    ## LTP to SCH
    def TangentPlane2TangentSphere(self, lat, lon=None, hdg=None):
        """TangentPlane2TangentSphere(self, lat, lon, hdg)

        Return a function of (x, y, z) that computes transformation from
        tangent plane to (s, c, h) coordiantes for input (lat, lon, hdg).
        """
        lat, lon, hdg = self._parse_peg(lat, lon, hdg)
        R = self.localRad(lat, hdg)
        def ltp2sch_wrapped(x, y, z):
            """Dynamically compiled function:
            function of     x, y, z  (meters)
            returns a tuple    s, c, h, (meters)
            """
            h   = R + z
            rho = (x**2 + y**2 + h**2)**0.5
            s = R*arctan(x/h)
            c = R*arcsin(y/rho)
            h = rho - R
            return s, c, h

#        ltp2sch.__doc__ += "\n On Ellipsoid "+self.model+"\n"
#        ltp2sch.__doc__ += "At latitide=%d and heading=%d"%(lat, hdg)
        return ltp2sch_wrapped

    ## A curried function that fixes peg point and returns a function
    ## from SCH to LTP
    def TangentSphere2TangentPlane(self, lat, lon=None, hdg=None):
        """TangenSphere2TangentPlane(self, lat, lon, hdg)

        Return a function of (s, c, h) that computes transformation from
        tangent sphere to (x, y, z) coordiantes for input (lat, lon, hdg).
        """
        lat, lon, hdg = self._parse_peg(lat, lon, hdg)
        R = self.localRad(lat, hdg)
        def sch2ltp_wrapped(s, c, h):
            """Dynamically compiled function:
            function of     s, c, h  (meters)
            returns a tuple    x, y, z, (meters)
            """
            s_lat = s/R
            c_lat = c/R
            r     = h + R

            P = euclid.polar2vector(
                Polar(r, np.pi/2 - c_lat, s_lat)
                )

            return P.y, P.z, P.x-R

#        sch2ltp.__doc__ += "\n On Ellipsoid "+self.model+"\n"
#        sch2ltp.__doc__ += "At latitide=%d and heading=%d"%(lat, hdg)
        return sch2ltp_wrapped

    ## Convience function for parsing arguments that might be a peg point.
    @staticmethod
    def _parse_peg(peg, lon, hdg):
        return (
            (peg.lat, peg.lon, peg.hdg)
            if lon is None else
            (peg, lon, hdg)
            )

## A OblateEllipsoid with coordinate system sub-classes attached
class Ellipsoid(_OblateEllipsoid, EllipsoidTransformations):
    """ellipsoid = Ellipsoid(a, finv [, model="Unknown"])

    a       singleton is the semi-major axis
    finv    is the inverse flatteing.
    model   string name of the ellipsoid

    The Ellipsoid is an oblate ellipsoid of revoltuion, and is alot more
    than 2-paramters

    See __init__.__doc__ for more.
    """
    ## The Ellipsoid needs access to the coordinates.PegPoint object
    PegPoint = coordinates.PegPoint

    def ECEF(self, x, y, z):
        return coordinates.ECEF(x,
                                y,
                                z,
                                ellipsoid=self)
    def LLH(self, lat, lon, hgt, peg_point=None):
        return coordinates.LLH(lat,
                               lon,
                               hgt,
                               ellipsoid=self)
    def LTP(self, x, y, z, peg_point=None):
        return coordinates.LTP(x,
                               y,
                               z,
                               peg_point=peg_point,
                               ellipsoid=self)
    def SCH(self, s, c, h, peg_point=None):
        return coordinates.SCH(s,
                               c,
                               h,
                               peg_point=peg_point,
                               ellipsoid=self)

## An iterative function that converts ECEF to LLH, given and
## ellipsoid.Ellipsoid instance \n It's really a method, but is broken out, so
## you can chose a different function without changing the class (TBD).
def ecef2llh_iterative(ellipsoid_of_revolution, x, y, z, iters=10):
    """ecef2llh(ellipsoid_of_revolution, x, y, z [,iters=10])--> (lat, lon, hgt)

    Input:
    ------
    ellipsoid_of_revolution  an Ellipsoid instance
    x
    y           ECEF coordiantes (singleton, or array, or whatever
    z

    KeyWord:
    --------
    iters      controls the number of iteration in the loop to compute the
               latitude.

    Ouput:
    -----
    lat       is the latitude (deg)
    lon       is the longitude (deg)
    h         is the heigh (m)
    """
    lon = arctan2(y,x) * 180/np.pi
    p = (x**2 + y**2)**0.5
    r = (x**2 + y**2 + z**2)**0.5

    phi = arctan2(p, z)
    while iters>0:
        RN = ellipsoid_of_revolution.N(phi*180/np.pi)
        h = (p/np.cos(phi)) - RN
        phi = arctan(
            (z/p)/(1-ellipsoid_of_revolution.e2 * RN/(RN+h))
            )
        iters -= 1
        pass

    phi *= 180/np.pi
    h = p/cosd(phi) - ellipsoid_of_revolution.N(phi)

    return (phi, lon, h)


## A good to the millimeter level function from Scott Hensely that does not use
## iteration-- good enough for L-Band.
def ecef2llh_noniterative(ellipsoid_of_revolution, x, y, z):
    return NotImplemented

## This function gets called by the Ellipsoid's method.
ecef2llh = ecef2llh_iterative

