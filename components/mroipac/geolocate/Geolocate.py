#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2010 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Walter Szeliga
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import os
import math
from isceobj.Location.Coordinate import Coordinate
from iscesys.Component.Component import Port, Component

class Geolocate(Component):
    
    logging_name = 'mroipac.geolocate'
    def __init__(self):
        super(Geolocate, self).__init__()
        # Ellipsoid information
        self.a = None
        self.e2 = None
        # Other information
        self.pos = []
        self.vel = []
        self.range = None
        self.squint = None
        return  None

    def createPorts(self):
        planetPort = Port(name='planet',method=self.addPlanet)
        self._inputPorts.add(planetPort)
        return None

    def addPlanet(self):
        planet = self._inputPorts.getPort('planet').getObject()
        try:
            self.a = planet.get_elp().get_a()
            self.e2 = planet.get_elp().get_e2()
        except AttributeError as strerr:
            self.logger.error(strerr)
            raise AttributeError
                
    def geolocate(self, position=None, velocity=None, range=None, squint=None, side=-1):
        """
        Given a position and velocity vector, along with a range and squint angle,
        return the geolocated coordinate and look angle from the satellite to the ground.

        @param position the cartesian position vector of the satellite [m]
        @param velocity the cartesian velocity vector of the satellite [m/s]
        @param range the range from the satellite to the ground [m]
        @param squint the squint angle of the satellite [radians]
        @param side the look side of the satellite [-1 for right, +1 for left]
        @return (\a tuple) coordinate object, look angle, incidence angle
        """
        from ctypes import cdll, c_double, c_int
        for port in self._inputPorts:
            method = port.getMethod()
            method()
            
        libgeo = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),'libgeolocate.so'))

        # inputs
        pos_c = (c_double * len(position))()
        pos_c[:] = position
        vel_c = (c_double * len(velocity))()
        vel_c[:] = velocity
        range_c = c_double(range)
        squint_c = c_double(squint)
        side_c = c_int(side)
        a_c = c_double(self.a)
        e2_c = c_double(self.e2)

        # outputs
        llh_c = (c_double*3)()
        lookAngle_c = (c_double*1)()
        incidenceAngle_c = (c_double*1)()

        # call to c wrapper to fortran subroutine
        # need to modify fortran subroutine to also return lookDirection
        libgeo.geolocate_wrapper(pos_c, vel_c, range_c, squint_c, side_c, a_c, e2_c, llh_c, lookAngle_c, incidenceAngle_c)

        # extract outputs
        # any issue with float versus double?
        coordinate = Coordinate()
        coordinate.setLatitude(math.degrees(llh_c[0]))
        coordinate.setLongitude(math.degrees(llh_c[1]))
        coordinate.setHeight(llh_c[2])
        lookAngle = math.degrees(lookAngle_c[0])
        incidenceAngle = math.degrees(incidenceAngle_c[0])

        # return outputs
        # proper syntax for return statement?
        return coordinate,lookAngle,incidenceAngle

