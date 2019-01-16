#!/usr/bin/env python3
import math
import collections
## Const Constants are now module constants
pi = math.pi
m = 1.0
kg = 1.0
s = 1.0
K = 1.0
rad = 1.0
km = 1000.0 *  m
hour = 3600.0 * s
day = 24.0 * hour
deg = (math.pi/180.0) * rad
Watt = 1.0 * kg * m**2 / s**3
G = 6.6742E-11 * m**3 /(kg * s**2)
AU = 1.49598E11 * m
c = 299792458.0 * m/s

## A namedtuple for ellipsoid parameters
ae2 = collections.namedtuple('SemiMajorAxisAndEccentricitySquared',
                             'a e2')
__todo__ = ('Cupid', 'Perdita')

## A metaclass for constant classes
class init_from_constants_type(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args)
        for name, value in kwargs.items():
            setattr(obj, name, value)
        return obj

    pass

## A class for constant instances
class ConstantClass(object, metaclass = init_from_constants_type):
    """obj = ConstantClass(**kwargs)

    makes an object supporting:

    obj[key] --> value

    or

    obj.key --> value

    for key, value in kwargs.items()

    The current configuration permits attribute manipulation, but
    forbids item setting or deleting-- however, that is strictly a matter
    of taste.
    """
#    __metaclass__ = init_from_constants_type
    ## Allow dictionary emulation
    def __getitem__(self, key):
        return getattr(self, key)
    def __delitem__(self, key):
        return self._raise('delete item')
    def __setitem__(self, key, value):
        return self._raise('set item')
    def _raise(self, message):
        raise TypeError(
            "cannon %s with a % s, use attribute access" % (
                message, self.__class__.__name__
                )
            )

    def __contains__(self, key):
        try:
            result = self[key]
            return True
        except (TypeError, AttributeError):
            return False
        pass

    def get(self, key, default=None):
        try:
            result = self[key]
        except AttributeError:
            result = default
            pass
        return result

    pass

## Keepin Const around for backward compatibility
Const = ConstantClass(
    pi = pi,
    m = m,
    kg = kg,
    s = s,
    K = K,
    rad = rad,
    km = km,
    hour = hour,
    day = day,
    deg = deg,
    Watt = Watt,
    G = G,
    AU = AU,
    c = c
    )

## Solar Data
SolarData = ConstantClass(
    rotationPeriod = 25.38 * day,
    equatorialRadius = 6.960E8 * m,
    GM = 1.989E30 * kg * G,
    luminosity = 3.826E26 * Watt,
    )

## Planet Data
PlanetsData = ConstantClass(
    ## Planet and some dwarf planet names
    names = ('Mercury',
             'Venus',
             'Earth',
             'Mars',
             'Jupiter',
             'Saturn',
             'Uranus',
             'Neptune',
             'Pluto'),
    # Dictionary of planet mean orbital distances in ms
    # from "Satellites" Burns and Matthews, editors,
    # University of Arizona Press, Tucson, 1986
    meanOrbitalDistance = ConstantClass(
        Mercury=     0.387 * AU,
        Venus=       0.723 * AU,
        Earth=       1.000 * AU,
        Mars=        1.524 * AU,
        Jupiter=     5.203 * AU,
        Saturn=      9.539 * AU,
        Uranus=     19.182 * AU,
        Neptune=    30.058 * AU,
        Pluto=      39.440 * AU
        ),
    # Dictionary of planets rotation periods in seconds
    # from "Satellites" Burns and Matthews, editors,
    # University of Arizona Press, Tucson, 1986
    rotationPeriod = ConstantClass(
        Mercury=   58.65   * day,
        Venus=    243.01   * day,
        Earth=     23.934472399 * hour,  #Based on mean angular velocity,
                                         #http://hpiers.obspm.fr/eop-pc/models/constants.html
        Mars=      24.6299 * hour,
        Jupiter=    9.841  * hour,
        Saturn=    10.233  * hour,
        Uranus=    17.3    * hour,
        Neptune=   18.2    * hour,
        Pluto=      6.387  * day
        ),
    obliquity = ConstantClass(
        Mercury=      2.   * deg,   # +- 3 * deg
        Venus=      177.3  * deg,
        Earth=       23.45 * deg,
        Mars=        23.98 * deg,
        Jupiter=      3.12 * deg,
        Saturn=      26.73 * deg,
        Uranus=      97.86 * deg,
        Neptune=     29.56 * deg,
        Pluto=      118.5  * deg
        ),
    effectiveBlackbodyTemperature = ConstantClass(
        Mercury=   442. * K,
        Venus=     244. * K,
        Earth=     253. * K,
        Mars=      216. * K,
        Jupiter=    87. * K,
        Saturn=     63. * K,
        Uranus=     33. * K,
        Neptune=    32. * K,
        Pluto=      43. * K
        ),
    # Dictionary of planet GM in meter**3/second**2
    # from "Satellites" Burns and Matthews, editors,
    # University of Arizona Press, Tucson, 1986
    GM = ConstantClass(
        Earth=    398600448073000.0 * m**3/s**2,    # Embedded in ROI_PAC
        Mercury=     0.3303E24 * kg * G,
        Venus=       4.8700E24 * kg * G,
#       Earth=       5.9767E24 * kg * G,   # GM is more well-determined than either G or M
        Mars=        0.6421E24 * kg * G,
        Jupiter=  1900.E24     * kg * G,
        Saturn=    568.8E24    * kg * G,
        Uranus=     86.87E24   * kg * G,
        Neptune=   102.0E24    * kg * G,
        Pluto=       0.013E24  * kg * G,
        ),
    # Dictionary of planet equatorial radius
    # from "Satellites" Burns and Matthews, editors,
    # University of Arizona Press, Tucson, 1986
    equatorialRadius = ConstantClass(
        Mercury=     2.439E6 * m,
        Venus=       6.051E6 * m,
        Earth=       6.378E6 * m,
        Mars=        3.393E6 * m,
        Jupiter=    71.398E6 * m,
        Saturn=     60.33E6  * m,
        Uranus=     26.20E6  * m,
        Neptune=    25.23E6  * m,
        Pluto=       1.5E6   * m
        ),
    density = ConstantClass(
        Mercury=   5.43E3  * kg/m**3,
        Venus=     5.25E3  * kg/m**3,
        Earth=     5.518E3 * kg/m**3,
        Mars=      3.95E3  * kg/m**3,
        Jupiter=   1.33E3  * kg/m**3,
        Saturn=    0.69E3  * kg/m**3,
        Uranus=    1.15E3  * kg/m**3,
        Neptune=   1.55E3  * kg/m**3,
        Pluto=     0.9E3   * kg/m**3
        ),
    # Dictionary of planet J2 --- coefficient of the second zonal harmonic in the expansion of the gravitational potential:
    #
    # V = (GM/r) * [ 1 - J2*(a/r)**2 * P2(cos(lat)) - J4*(a/r)**4 * P4(cos(lat)) - ... ]
    #
    # where a is the semi-major axis of the ellipsoid and lat is the spherical coordinate colatitude (not geodetic latitude.
    # J2 is related to the oblateness (or flattening) of the Earth relaitve to a sphere.  For a uniform ellipsoid,
    #
    # J2 = (2/3)*f - (1/3)*m - (1/3)*f**2 + (2/21)*f*m
    # J4 = -(4/5)*f**2 + (4/7)*f*m
    #
    # f = flattening = (b-a)/a
    # m = (omega*a)**2*b/GM = ratio of centrifugal to gravitational forces at equator
    # a = semi-major axis (equatorial radius)
    # b = semi-minor axis (polar radius)
    # omega = frequency of planet's spin
    #
    # http://www.ngs.noaa.gov/PUBS_LIB/Geodesy4Layman/TR80003F.HTM
    #
    J2 = ConstantClass(
        Mercury=     8.E-5,      # +- 6E-5
        Venus=       0.6E-5,
        Earth=     108.3E-5,
        Mars=      196.0E-5,     # +-1.8E-5
        Jupiter=  1473.6E-5,     # +-0.1E-5
        Saturn=   1667.E-5,      # +-3.E-5
        Uranus=    333.9E-5,     # +-0.3E-5
        Neptune=   430.E-5       # +-30.E-5
        ),
    J4 = ConstantClass(
        Earth=    -0.16E-5,
        Mars=     -3.2E-5,    # +- 0.7E-5
        Jupiter= -58.7E-5,    # +- 0.5E-7
        Saturn= -103.E-5,     # +- 7E-5
        Uranus=   -3.2E-5,    # +- 0.4E-5
        ),
    # Dictionary of tuples of (semi-major axis in meter,eccentricity-squared)
    # Unless the planet has more than one ellipsoid model (Earth) - Dictionary of Dictionary.
    ellipsoid = ConstantClass(
        Earth={
            'Airy-1830':                   ae2(6377563.396*m, 0.0066705400000),
            'Modified-Airy':               ae2(6377340.189*m, 0.0066705400000),
            'Australian':                  ae2(6378160.000*m, 0.0066945418546),
            'Bessel-1841-Namibia':         ae2(6377483.865*m, 0.0066743722318),
            'Bessel-1841':                 ae2(6377397.155*m, 0.0066743722318),
            'Clarke-1866':                 ae2(6378206.400*m, 0.0067686579976),
            'Clarke-1880':                 ae2(6378249.145*m, 0.0068035112828),
            'Everest-India-1830':          ae2(6377276.345*m, 0.0066378466302),
            'Everest-Sabah-Sarawak':       ae2(6377298.556*m, 0.0066378466302),
            'Everest-India-1956':          ae2(6377301.243*m, 0.0066378466302),
            'Everest-Malaysia-1969':       ae2(6377295.664*m, 0.0066378466302),
            'Everest-Malay-Singapore-1948':ae2(6377304.063*m, 0.0066378466302),
            'Everest-Pakistan':            ae2(6377309.613*m, 0.0066378466302),
            'Modified-Fischer-1960':       ae2(6378155.000*m, 0.0066934216230),
            'Helmert-1906':                ae2(6378200.000*m, 0.0066934216230),
            'Hough-1960':                  ae2(6378270.000*m, 0.0067226700223),
            'Indonesian-1974':             ae2(6378160.000*m, 0.0066946090804),
            'International-1924':          ae2(6378388.000*m, 0.0067226700223),
            'Krassovsky-1940':             ae2(6378245.000*m, 0.0066934216230),
            'GRS-80':                      ae2(6378137.000*m, 0.0066943800229),
            'South-American-1969':         ae2(6378160.000*m, 0.0066945418546),
            'WGS-72':                      ae2(6378135.000*m, 0.0066943177783),
            'WGS-84':                      ae2(6378137.000*m, 0.0066943799901)
            },
        ),
    satellites = ConstantClass(
        Earth=    ("Moon",),
        Mars=     ("Phobos",
                   "Deimos"),
        Jupiter=  ("Metis",
                   "Adrastea",
                   "Amalthea",
                   "Thebe",
                   "Io",
                   "Europa",
                   "Ganymede",
                   "Callisto",
                   "Leda",
                   "Himalia",
                   "Lysithea",
                   "Elara",
                   "Ananke",
                   "Carme",
                   "Pasiphae",
                   "Sinope",
                   "Halo",
                   "Main_Ring",
                   "Gossamer_Ring"),
        Saturn=   ("Atlas",
                   "Prometheus",
                   "Pandora",
                   "Epimetheus",
                   "Janus",
                   "Mimas",
                   "Enceladus",
                   "Tethys",
                   "Telesto",
                   "Calypso",
                   "Dione",
                   "Helene",
                   "Rhea",
                   "Titan",
                   "Hyperion",
                   "Iapetus",
                   "Phoebe",
                   "D_Ring",
                   "C_Ring",
                   "B_Ring",
                   "A_Ring",
                   "F_Ring",
                   "G_Ring",
                   "E_Ring"),
        Uranus=   ("Cordelia",
                   "Ophelia",
                   "Bianca",
                   "Cressida",
                   "Desdemona",
                   "Juliet",
                   "Portia",
                   "Rosalind",
                   "Belinda",
                   "Puck",
                   "Miranda",
                   "Ariel",
                   "Umbriel",
                   "Titania",
                   "Oberon",
                   "Rings"),
        Neptune=  ("Triton",
                   "Nereid",
                   "Ring Arc"),
        Pluto=    ("Charon",)
        )
    )


## temporary constants
_orbitalPeriod = {
    'Moon':             27.3217   * day,
    'Phobos':            0.319    * day,
    'Deimos':            1.263    * day,
    'Metis':             0.2948   * day,
    'Adrastea':          0.2983   * day,
    'Amalthea':          0.4981   * day,
    'Thebe':             0.6745   * day,
    'Io':                1.769    * day,
    'Europa':            3.551    * day,
    'Ganymede':          7.155    * day,
    'Callisto':         16.689    * day,
    'Leda':            238.72     * day,
    'Himalia':         250.57     * day,
    'Lysithea':        259.22     * day,
    'Elara':           259.65     * day,
    'Ananke':         -631.       * day,
    'Carme':          -692.       * day,
    'Pasiphae':       -735.       * day,
    'Sinope':         -758.       * day,
    'Atlas':             0.602    * day,
    'Prometheus':        0.613    * day,
    'Pandora':           0.629    * day,
    'Epimetheus':        0.694    * day,
    'Janus':             0.695    * day,
    'Mimas':             0.942    * day,
    'Enceladus':         1.370    * day,
    'Tethys':            1.888    * day,
    'Telesto':           1.888    * day,
    'Calypso':           1.888    * day,
    'Dione':             2.737    * day,
    'Helene':            2.737    * day,
    'Rhea':              4.518    * day,
    'Titan':            15.945    * day,
    'Hyperion':         21.277    * day,
    'Iapetus':          79.331    * day,
    'Phoebe':         -550.48     * day,
    'Cordelia':            0.336    * day,
    'Ophelia':            0.377    * day,
    'Bianca':            0.435    * day,
    'Cressida':            0.465    * day,
    'Desdemona':            0.476    * day,
    'Juliet':            0.494    * day,
    'Portia':            0.515    * day,
    'Rosalind':            0.560    * day,
    'Belinda':            0.624    * day,
    'Puck':            0.764    * day,
    'Miranda':           1.413    * day,
    'Ariel':             2.520    * day,
    'Umbriel':           4.144    * day,
    'Titania':           8.706    * day,
    'Oberon':           13.463    * day,
    'Triton':           -5.877    * day,
    'Nereid':          360.16     * day,
    'Charon':            6.387    * day
    }


SatellitesData=ConstantClass(
    planet = ConstantClass(
        Moon=       'Earth',
        Phobos=     'Mars',
        Deimos=     'Mars',
        Metis=      'Jupiter',
        Adrastea=   'Jupiter',
        Amalthea=   'Jupiter',
        Thebe=      'Jupiter',
        Io=         'Jupiter',
        Europa=     'Jupiter',
        Ganymede=   'Jupiter',
        Callisto=   'Jupiter',
        Leda=       'Jupiter',
        Himalia=    'Jupiter',
        Lysithea=   'Jupiter',
        Elara=      'Jupiter',
        Ananke=     'Jupiter',
        Carme=      'Jupiter',
        Pasiphae=   'Jupiter',
        Sinope=     'Jupiter',
        Halo=       'Jupiter',
        Main_Ring=  'Jupiter',
        Gossamer_Ring=  'Jupiter',
        Atlas=          'Saturn',
        Prometheus=     'Saturn',
        Pandora=        'Saturn',
        Epimetheus=     'Saturn',
        Janus=          'Saturn',
        Mimas=          'Saturn',
        Enceladus=      'Saturn',
        Tethys=         'Saturn',
        Telesto=        'Saturn',
        Calypso=        'Saturn',
        Dione=          'Saturn',
        Helene=         'Saturn',
        Rhea=           'Saturn',
        Titan=          'Saturn',
        Hyperion=       'Saturn',
        Iapetus=        'Saturn',
        Phoebe=         'Saturn',
        D_Ring=         'Saturn',
        C_Ring=         'Saturn',
        B_Ring=         'Saturn',
        A_Ring=         'Saturn',
        F_Ring=         'Saturn',
        G_Ring=         'Saturn',
        E_Ring=         'Saturn',
        Cordelia=         'Uranus',
        Ophelia=         'Uranus',
        Bianca=         'Uranus',
        Cressida=         'Uranus',
        Desdemona=         'Uranus',
        Juliet=         'Uranus',
        Portia=         'Uranus',
        Rosalind=         'Uranus',
        Belinda=         'Uranus',
        Puck=         'Uranus',
        Miranda=        'Uranus',
        Ariel=          'Uranus',
        Umbriel=        'Uranus',
        Titania=        'Uranus',
        Oberon=         'Uranus',
        Rings=          'Uranus',
        Triton=         'Neptune',
        Nereid=         'Neptune',
        Ring_Arc=       'Neptune',
        Charon=         'Pluto'
        ),
    orbitalSemimajorAxis = ConstantClass(
        Moon=            384.4E6    * m,
        Phobos=            9.378E6  * m,
        Deimos=           23.459E6  * m,
        Metis=           127.96E6   * m,
        Adrastea=        128.98E6   * m,
        Amalthea=        181.3E6    * m,
        Thebe=           221.90E6   * m,
        Io=              421.6E6    * m,
        Europa=          670.9E6    * m,
        Ganymede=       1070.E6     * m,
        Callisto=       1883.E6     * m,
        Leda=          11094.E6     * m,
        Himalia=       11480.E6     * m,
        Lysithea=      11720.E6     * m,
        Elara=         11737.E6     * m,
        Ananke=        21200.E6     * m,
        Carme=         22600.E6     * m,
        Pasiphae=      23500.E6     * m,
        Sinope=        23700.E6     * m,
        Atlas=           137.64E6   * m,
        Prometheus=      139.35E6   * m,
        Pandora=         141.70E6   * m,
        Epimetheus=      151.422E6  * m,
        Janus=           151.472E6  * m,
        Mimas=           185.52E6   * m,
        Enceladus=       238.02E6   * m,
        Tethys=          294.66E6   * m,
        Telesto=         294.66E6   * m,
        Calypso=         294.66E6   * m,
        Dione=           377.40E6   * m,
        Helene=          377.40E6   * m,
        Rhea=            527.04E6   * m,
        Titan=          1221.85E6   * m,
        Hyperion=       1481.1E6    * m,
        Iapetus=        3561.3E6    * m,
        Phoebe=        12952.E6     * m,
        Cordelia=           49.75E6   * m,
        Ophelia=           53.77E6   * m,
        Bianca=           59.16E6   * m,
        Cressida=           61.77E6   * m,
        Desdemona=           62.65E6   * m,
        Juliet=           64.63E6   * m,
        Portia=           66.10E6   * m,
        Rosalind=           69.93E6   * m,
        Belinda=           75.25E6   * m,
        Puck=           86.00E6   * m,
        Miranda=         129.8E6    * m,
        Ariel=           191.2E6    * m,
        Umbriel=         266.0E6    * m,
        Titania=         435.8E6    * m,
        Oberon=          582.6E6    * m,
        Triton=          354.3E6    * m,
        Nereid=          551.5E6    * m,
        Charon=           19.1E6    * m
        ),
    orbitalPeriod = _orbitalPeriod,
    rotationPeriod = ConstantClass(
        Moon=             _orbitalPeriod['Moon'],
        Phobos=           _orbitalPeriod['Phobos'],
        Deimos=           _orbitalPeriod['Deimos'],
        Amalthea=         _orbitalPeriod['Amalthea'],
        Io=               _orbitalPeriod['Io'],
        Europa=           _orbitalPeriod['Europa'],
        Ganymede=         _orbitalPeriod['Ganymede'],
        Callisto=         _orbitalPeriod['Callisto'],
        Himalia=          0.4 * day,
        Epimetheus=       _orbitalPeriod['Epimetheus'],
        Janus=            _orbitalPeriod['Janus'],
        Mimas=            _orbitalPeriod['Mimas'],
        Enceladus=        _orbitalPeriod['Enceladus'],
        Tethys=           _orbitalPeriod['Tethys'],
        Dione=            _orbitalPeriod['Dione'],
        Rhea=             _orbitalPeriod['Rhea'],
        Iapetus=          _orbitalPeriod['Iapetus'],
        Phoebe=           0.4 * day,
        Miranda=          _orbitalPeriod['Miranda'],
        Ariel=            _orbitalPeriod['Ariel'],
        Umbriel=          _orbitalPeriod['Umbriel'],
        Titania=          _orbitalPeriod['Titania'],
        Oberon=           _orbitalPeriod['Oberon'],
        Triton=           _orbitalPeriod['Triton'],
        ),
    orbitalEccentricity = ConstantClass(
        Moon=             0.05490,
        Phobos=           0.015,
        Deimos=           0.00052,
        Metis=            0.,       # < 0.004
        Adrastea=         0.,
        Amalthea=         0.003,
        Thebe=            0.015,    # +- 0.006
        Io=               0.0041,
        Europa=           0.0101,
        Ganymede=         0.0006,
        Callisto=         0.007,
        Leda=             0.148,
        Himalia=          0.158,
        Lysithea=         0.107,
        Elara=            0.207,
        Ananke=           0.169,
        Carme=            0.207,
        Pasiphae=         0.378,
        Sinope=           0.275,
        Atlas=            0.,
        Prometheus=       0.0024,  # +- 0.0006
        Pandora=          0.0042,  # +- 0.0006
        Epimetheus=       0.009,      # +- 0.002
        Janus=            0.007,      # +- 0.002
        Mimas=            0.0202,
        Enceladus=        0.0045,
        Tethys=           0.0000,
        Telesto=          0.,
        Calypso=          0.,
        Dione=            0.0022,
        Helene=           0.005,
        Rhea=             0.0010,
        Titan=            0.0292,
        Hyperion=         0.1042,
        Iapetus=          0.0283,
        Phoebe=           0.163,
        Cordelia=           0.,
        Ophelia=           0.,
        Bianca=           0.,
        Cressida=           0.,
        Desdemona=           0.,
        Juliet=           0.,
        Portia=           0.,
        Rosalind=           0.,
        Belinda=           0.,
        Puck=           0.,
        Miranda=          0.0027,
        Ariel=            0.0034,
        Umbriel=          0.0050,
        Titania=          0.0022,
        Oberon=           0.0008,
        Triton=           0.,        # < 0.0005
        Nereid=           0.75,
        Charon=           0.
        ),
    orbitalInclination = ConstantClass(
        Moon=             5.15  * deg,
        Phobos=           1.02  * deg,
        Deimos=           1.82  * deg,
        Metis=            0.    * deg,
        Adrastea=         0.    * deg,
        Amalthea=         0.40  * deg,
        Thebe=            0.8   * deg,  # +- 0.2 * deg
        Io=               0.040 * deg,
        Europa=           0.470 * deg,
        Ganymede=         0.195 * deg,
        Callisto=         0.281 * deg,
        Leda=            27.    * deg,
        Himalia=         28.    * deg,
        Lysithea=        29.    * deg,
        Elara=           28.    * deg,
        Ananke=         147.    * deg,
        Carme=          163.    * deg,
        Pasiphae=       148.    * deg,
        Sinope=         153.    * deg,
        Atlas=            0.    * deg,
        Prometheus=       0.0   * deg,     # +- 0.1
        Pandora=          0.0   * deg,     # +- 0.1
        Epimetheus=       0.34  * deg,     # +- 0.05
        Janus=            0.14  * deg,     # +- 0.05
        Mimas=            1.53  * deg,
        Enceladus=        0.02  * deg,
        Tethys=           1.09  * deg,
        Telesto=          0.    * deg,
        Calypso=          0.    * deg,
        Dione=            0.02  * deg,
        Helene=           0.2   * deg,
        Rhea=             0.35  * deg,
        Titan=            0.33  * deg,
        Hyperion=         0.43  * deg,
        Iapetus=          7.52  * deg,
        Phoebe=         175.3   * deg,
        Cordelia=           0.    * deg,
        Ophelia=           0.    * deg,
        Bianca=           0.    * deg,
        Cressida=           0.    * deg,
        Desdemona=           0.    * deg,
        Juliet=           0.    * deg,
        Portia=           0.    * deg,
        Rosalind=           0.    * deg,
        Belinda=           0.    * deg,
        Puck=           0.    * deg,
        Miranda=          4.22  * deg,
        Ariel=            0.31  * deg,
        Umbriel=          0.36  * deg,
        Titania=          0.14  * deg,
        Oberon=           0.10  * deg,
        Triton=         159.0   * deg,        # +- 1.5 * deg
        Nereid=          27.6   * deg,
        Charon=          94.3   * deg         # +- 1.5 * deg
        ),
    radius = ConstantClass(
        Moon=             1738.  * km,
        Ganymede=         2631.  * km,  # +- 10 * km
        Callisto=         2400.  * km,  # +- 10 * km
        Io=               1815   * km,  # +-  5 * km
        Europa=           1569.  * km,  # +- 10 * km
        Leda=                8.  * km,  # approximate
        Himalia=            90.  * km,  # +- 10 * km
        Lysithea=           20.  * km,  # approximate
        Elara=              40.  * km,  # +-  5 * km
        Ananke=             15.  * km,  # approximate
        Carme=              22.  * km,  # approximate
        Pasiphae=           35.  * km,  # approximate
        Sinope=             20.  * km,  # approximate
        Titan=            2575.  * km,  # +- 2
        Rhea=              764.  * km,  # +- 4
        Iapetus=           718.  * km,  # +- 8
        Dione=             559.  * km,  # +- 5
        Tethys=            524.  * km,  # +- 5
        Enceladus=         251.  * km,  # +- 5
        Mimas=             197.  * km,  # +- 3
        Titania=           800.  * km,  # +-  5
        Oberon=            775.  * km,  # +- 10
        Umbriel=           595.  * km,  # +- 10
        Ariel=             580.  * km,  # +-  5
        Miranda=           242.  * km,  # +-  5
        Puck=             40.  * km,
        Portia=             40.  * km,
        Juliet=             30.  * km,
        Cressida=             30.  * km,
        Rosalind=             30.  * km,
        Belinda=             30.  * km,
        Desdemona=             30.  * km,
        Cordelia=             25.  * km,
        Ophelia=             25.  * km,
        Bianca=             25.  * km,
        Triton=           1750.  * km,      # +- 250.
        Nereid=            200.  * km,
        Charon=             500.  * km
        ),
    mass = ConstantClass(
        Moon=              734.9E20  * kg,
        Phobos=              1.26E16 * kg,
        Deimos=              1.8E15  * kg,
        Ganymede=         1482.3E20  * kg,
        Callisto=         1076.6E20  * kg,
        Io=                894.E20   * kg,
        Europa=            480.E20   * kg,
        Titan=            1345.7E20  * kg,
        Rhea=               24.9E20  * kg,
        Iapetus=            18.8E20  * kg,
        Dione=              10.5E20  * kg,
        Tethys=              7.6E20  * kg,
        Enceladus=           0.8E20  * kg,
        Mimas=               0.38E20 * kg,
        Titania=            34.3E20  * kg,
        Oberon=             28.7E20  * kg,
        Umbriel=            11.8E20  * kg,
        Ariel=              14.4E20  * kg,
        Miranda=             0.71E20 * kg,
        Triton=           1300.E20   * kg,
        )
    )



## All the satellite news that's fit to print
SatelliteFactSheet = collections.namedtuple(
    'SatelliteFactSheet',
    " ".join(SatellitesData.__dict__.keys())
    )

def get_satellite_fact_sheet(name):
    """input a string name of a satellite, and get its SatelliteFactSheet"""
    from operator import methodcaller
    return SatelliteFactSheet(
        *map(methodcaller('get', name),
             SatellitesData.__dict__.itervalues())
         )


## All the planet news that's fit to print
PlanetFactSheet = collections.namedtuple(
    'PlanetFactSheet',
    " ".join(filter(lambda x: x != 'names', PlanetsData.__dict__.keys()))
    )

def get_planet_fact_sheet(name):
    """input a string name of a planet, and get its PlanetFactSheet"""
    from operator import methodcaller
    return PlanetFactSheet(
        *map(methodcaller('get', name),
             filter(lambda x: not isinstance(x, tuple), PlanetsData.__dict__.itervalues())
             )
         )


del _orbitalPeriod
