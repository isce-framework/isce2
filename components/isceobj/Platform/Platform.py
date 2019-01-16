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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



import math
from iscesys.Component.Component import Component
from isceobj.Planet.Planet import Planet
from isceobj.Util.decorators import type_check

PLANET = Component.Facility(
    '_planet',
    public_name='PLANET',
    module='isceobj.Planet',
    factory='createPlanet',
    args=('Earth',),
    mandatory=False,
    doc="Planet factory"
)

SPACECRAFT_NAME = Component.Parameter('spacecraftName',
    public_name='SPACECRAFT_NAME',
    default=None,
    type = str,
    mandatory = True,
    doc = 'Name of the space craft')

MISSION = Component.Parameter('_mission',
    public_name='MISSION',
    default=None,
    type = str,
    mandatory = True,
    doc = 'Mission name')

ANTENNA_LENGTH = Component.Parameter('antennaLength',
    public_name='ANTENNA_LENGTH',
    default=None,
    type = float,
    mandatory = True,
    doc = 'Length of the antenna')

POINTING_DIRECTION = Component.Parameter('pointingDirection',
    public_name='POINTING_DIRECTION',
    default=None,
    type = int,
    mandatory = True,
    doc = '-1 for RIGHT, 1 for LEFT')

##
# This class allows the creation of a Platform object. The parameters that need to be set are
#\verbatim
#PLANET: Name of the planet about which the platform orbits. Mandatory.
#SPACECRAFT_NAME: Name of the spacecraft. Mandatory.
#BODY_FIXED_VELOCITY:
#SPACECRAFT_HEIGHT: Height of the sapcecraft. Mandatory.
#POINTING_DIRECTION: 
#ANTENNA_LENGTH: Length of the antenna. Mandatory.
#ANTENNA_SCH_VELOCITY
#ANTENNA_SCH_ACCELERATION
#HEIGHT_DT
#\endverbatim
#Since the Platform class inherits the Component.Component, the methods of initialization described in the Component package can be used.
#Moreover each parameter can be set with the corresponding accessor method setParameter() (see the class member methods).
class Platform(Component):

    family = 'platform'
    logging_name = 'isce.isceobj.platform'

    parameter_list = (
                      SPACECRAFT_NAME,
                      MISSION,
                      ANTENNA_LENGTH,
                      POINTING_DIRECTION)
    
    facility_list = (
                      PLANET,
                      )
    
    def __init__(self, name=''):
        super(Platform, self).__init__(family=self.__class__.family, name=name)
        return None

    def setSpacecraftName(self,var):
        self.spacecraftName = str(var)
        return
        
    def setAntennaLength(self,var):
        self.antennaLength = float(var)
        return
    
    def setPointingDirection(self,var):
        self.pointingDirection = int(var)
        return
    
    def setMission(self,mission):
        self._mission = mission
        
    def getMission(self):
        return self._mission
    
    def getSpacecraftName(self):
        return self.spacecraftName or self._mission

    def getAntennaLength(self):
        return self.antennaLength
    
    def getPlanet(self):
        return self._planet
    
    @type_check(Planet)
    def setPlanet(self,planet):
        self._planet = planet
        return None
        
    planet = property(getPlanet, setPlanet)

    def __str__(self):
        retstr = "Mission: (%s)\n"
        retlst = (self._mission,)
        retstr += "Look Direction: (%s)\n"
        retlst += (self.pointingDirection,)
        retstr += "Antenna Length: (%s)\n"
        retlst += (self.antennaLength,)
        return retstr % retlst
    

class Orientation(Component):
    """A class for holding platform orientation information, such as squint
    angle and platform height"""

    dictionaryOfVariables = {'BODY_FIXED_VELOCITY' :
                                 ['self.bodyFixedVelocity', 'float',True],
                             'ANTENNA_SCH_VELOCITY' :
                                 ['self.antennaSCHVelocity','float',True],
                             'ANTENNA_SCH_ACCELERATION' :
                                 ['self.antennaSCHAcceleration','float',True]}    
    
    def __init__(self):
        super(Orientation, self).__init__()
        self.antennaSCHVelocity = []
        self.antennaSCHAcceleration = []        
        self.bodyFixedVelocity = None        
        self.pointingDirection = None
        self.descriptionOfVariables = {}
        return None
        
    def setSpacecraftHeight(self, var):
        self.spacecraftHeight = float(var)
    
    def getSpacecraftHeight(self):
        return self.spacecraftHeight
    
    def setBodyFixedVelocity(self, var):
        self.bodyFixedVelocity = float(var)
        return
    
    def setAntennaSCHVelocity(self, var):
        self.antennaSCHVelocity = var
        return

    def setAntennaSCHAcceleration(self, var):
        self.antennaSCHAcceleration = var
        return


def createPlatform():
    return Platform()
