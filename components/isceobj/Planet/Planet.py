#!/usr/bin/env python3
from __future__ import print_function
import math
from iscesys.Component.Component import Component
import isceobj.Planet.AstronomicalHandbook as AstronomicalHandbook
from isceobj.Planet.Ellipsoid import Ellipsoid
from iscesys.Component.Configurable import SELF

PNAME = Component.Parameter(
    'pname',
    public_name='PNAME',
    default='Earth',
    type=str,
    mandatory=True,
    intent='input',
    doc='Planet name'
)
ELLIPSOID_MODEL = Component.Parameter(
    'ellipsoidModel',
    public_name='ELLIPSOID_MODEL',
    default=None,
    type=str,
    mandatory=False,
    intent='input',
    doc='Ellipsoid model'
)

class Planet(Component):
    """
    A class to represent a planet.
    The parameters maintained internally are the following:

    elp = an ellipsoid model of class Ellipsoid

    GM  = Planet mass in units of acceleration * distance**2 --- 
    dividing by distance**2 from the center of the planet gives the
    gravitational acceleration at that distance and 
    dividing by the distance gives the gravitational potential field
    monopole term at that distance

    spin = radian frequency of the planet's spin
    """
    parameter_list = (
                      PNAME,
                      ELLIPSOID_MODEL
                     )
    
    family = 'planet'

    #modified the constructor so it takes the ellipsoid model. this way it
    #does not to be hardcoded to WGS-84. 
    #also ellipsoid as been modified so it has the model attribute
    def __init__(self,family='', name='',pname='', ellipsoidModel=None):
        
        super(Planet, self).__init__(family if family else  self.__class__.family, name=name)

        self._pname = pname
        self._ellipsoidModel = ellipsoidModel
        #Before all the initialization done in _configure was done here but now we want that
        #to be triggered also during the initialization of Configurable. By putting it into
        # _configure() we reach the goal 
        #Call configure() for backward compatibility. 
        self._configure()
        return None
    
    #put all the initialization
    def _configure(self):
        if self._ellipsoidModel is None:
            if self._pname == 'Earth':
                self._ellipsoidModel = 'WGS-84'
            else:
                self._ellipsoidModel = 'default'
                ########## TO BE DONE in AstronomicalHandbook.py:
                # define a generic model called
                # default that just maps the name of the planet to the corresponding
                # axis and eccentricity 
                #######################
                print(
                    'At the moment  there is no default ellipsoid defined for the planet',
                    self._pname)
                raise NotImplementedError
            pass
        if self._pname in AstronomicalHandbook.PlanetsData.names:
            self._ellipsoid = (
                Ellipsoid(
                    a=AstronomicalHandbook.PlanetsData.ellipsoid[
                        self._pname
                        ][self._ellipsoidModel].a,e2=AstronomicalHandbook.PlanetsData.ellipsoid[
                        self._pname
                        ][self._ellipsoidModel].e2,
                     model=self._ellipsoidModel)
                )
            self.GM = AstronomicalHandbook.PlanetsData.GM[self._pname]
            self.spin = (
                2.0*math.pi/
                AstronomicalHandbook.PlanetsData.rotationPeriod[self._pname]
                )
        else:
            self._ellipsoid = Ellipsoid()
            self.GM = 1.0
            self.spin = 1.0
            pass
    @property
    def pname(self):
        """Name of the planet"""
        return self._pname
    @pname.setter
    def pname(self, pname):
        self._pname = pname
        return None

    def set_name(self,pname):
        if not isinstance(pname,basestring):
            raise ValueError("attempt to instantiate a planet with a name %s that is not a string" % pname)
        self.pname = pname
        return None
    
    def get_name(self):
        return self.pname

    @property
    def ellipsoid(self):
        """Ellipsoid model of the planet.  See Ellipsoid class."""        
        return self._ellipsoid
    @ellipsoid.setter
    def ellipsoid(self, elp):
        self._ellipsoid = elp
        return None
    
    def get_elp(self):
        return self.ellipsoid

    @property
    def GM(self):
        """Mass of planet times Newton's gravitational constant in m**3/s**2"""
        return self._GM
    @GM.setter
    def GM(self, GM):
        try:
            self._GM = float(GM)
        except (TypeError, ValueError):
            raise ValueError(
                "invalid use of non-numeric object %s to set GM value "
                %
                str(GM)
                ) 
        return None
    
    def get_GM(self):
        return self.GM

    def set_GM(self, GM):
        self.GM = GM
        pass

    @property
    def spin(self):
        return self._spin
    @spin.setter
    def spin(self, spin):
        try:
            self._spin = float(spin)
        except (ValueError, TypeError):
            raise ValueError(
                "invalid use of non-numeric object %s to set spin " % spin
                )
        pass
        
    def get_spin(self):
        return self.spin
    
    def set_spin(self, spin):
        self.spin = spin

    @property
    def polar_axis(self):
        return self._polar_axis
    @polar_axis.setter
    def polar_axis(self, vector):
        """Give me a vector that is parallel to my spin axis"""
        from isceobj.Util.geo.euclid import Vector
        if not isinstance(vector, Vector):
            try:
                vector = Vector(*vector)
            except Exception:
                raise ValueError(
                    "polar axis must a Vector or length 3 container"
                    )
            pass
        self._polar_axis = vector.hat()
        return None

    @property
    def ortho_axis(self):
        return self._ortho_axis

    @property
    def primary_axis(self):
        return self._primary_axis

    @primary_axis.setter
    def primary_axis(self, vector):
        """Give me a vector in your coordinates that is orthogonal to my polar
        axis"""
        from isceobj.Util.geo.euclid import Vector
        if not isinstance(vector, Vector):
            try:
                vector = Vector(*vector)
            except Exception:
                raise ValueError(
                    "primary axis must a Vector or length 3 container"
                    )
            pass
        self._primary_axis = vector.hat()

        try:
            if self.polar_axis*self._primary_axis > 1.e-10:
                raise ValueError(
                    "polar_axis and primary_axis are not orthogonal"
                    )
        except AttributeError:
            class RaceHazard(Exception):
                """The outer class has methods that must be called in order.
                Should you fail to do so, this Exception shall be raised"""
                pass
            raise RuntimeError("You must set planet's polar axis first")
        
        self._ortho_axis = self.primary_axis.cross(self.polar_axis)
        pass
    pass

  

