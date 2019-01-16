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



import math
import datetime
import logging
from iscesys.Component.Component import Component
from isceobj.Util.decorators import type_check, pickled, logged


## This class stores platform pitch, roll and yaw information.
TIME = Component.Parameter(
    '_time',
    public_name='TIME',
    default=0,
    type=datetime.datetime,
    mandatory=True,
    intent='input',
    doc=''
)


PITCH = Component.Parameter(
    '_pitch',
    public_name='PITCH',
    default=0,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


ROLL = Component.Parameter(
    '_roll',
    public_name='ROLL',
    default=0,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


YAW = Component.Parameter(
    '_yaw',
    public_name='YAW',
    default=0,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


class StateVector(Component):


    parameter_list = (
                      TIME,
                      PITCH,
                      ROLL,
                      YAW
                     )




    family = 'attitudestatevector'

    def __init__(self,family='',name='', time=None, pitch=None, roll=None, yaw=None):
        super(StateVector, self).__init__(family if family else  self.__class__.family, name=name)
        self._time = time
        self._pitch = pitch
        self._roll = roll
        self._yaw = yaw

        return None

    def toList(self):
        return [self._time.strftime('%Y-%m-%dT%H:%M:%S.%f'),self._pitch,self._roll,self._yaw]


    @type_check(datetime.datetime)
    def setTime(self, time):
        self._time = time
        pass

    def getTime(self):
        return self._time

    def setPitch(self, pitch):
        self._pitch = pitch

    def getPitch(self):
        return self._pitch

    def setRoll(self, roll):
        self._roll = roll

    def getRoll(self):
        return self._roll

    def setYaw(self, yaw):
        self._yaw = yaw

    def getYaw(self):
        return self._yaw

    def __str__(self):
        retstr = "Time: %s\n"
        retlst = (self.time,)
        retstr += "Pitch: %s\n"
        retlst += (self.pitch,)
        retstr += "Roll: %s\n"
        retlst += (self.roll,)
        retstr += "Yaw: %s\n"
        retlst += (self.yaw,)
        return retstr % retlst

    time = property(getTime, setTime)
    pitch = property(getPitch, setPitch)
    roll = property(getRoll, setRoll)
    yaw = property(getYaw, setYaw)
    pass


ATTITUDE_SOURCE = Component.Parameter(
    '_attitudeSource',
    public_name='ATTITUDE_SOURCE',
    default=None,
    type=str,
    mandatory=False,
    intent='input',
    doc=''
)

STATE_VECTORS = Component.Parameter(
    '_stateVectors',
    public_name='STATE_VECTORS',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    intent='input',
    doc=''
)


ATTITUDE_QUALITY = Component.Parameter(
    '_attitudeQuality',
    public_name='ATTITUDE_QUALITY',
    default=None,
    type=str,
    mandatory=False,
    intent='input',
    doc=''
)

## This class encapsulates spacecraft attitude information
## The Attitude class consists of a list of  StateVector objects
## and provides an iterator over this list.
@pickled
class Attitude(Component):


    parameter_list = (
                      ATTITUDE_SOURCE,
                      STATE_VECTORS,
                      ATTITUDE_QUALITY
                     )


    logging_name = 'isce.Attitude'
    min_length_for_interpolation = 3

    family = 'attitude'

    def __init__(self,family='',name=''):
        self._minTime = datetime.datetime(year=datetime.MAXYEAR,
                                          month=12,
                                          day=31)
        self._maxTime = datetime.datetime(year=datetime.MINYEAR,
                                          month=1,
                                          day=1)
        self._last = 0
        self.cpStateVectors = []
        super(Attitude, self).__init__(family if family else  self.__class__.family, name=name)

        return None

    def adaptToRender(self):

        self._cpStateVectors = []
        svList = []
        for sv in self.stateVectors:
            svList.append(sv.toList())
            #for some reason the deepcopy failed so adding one vector at the time
            self._cpStateVectors.append(sv)
        self.stateVectors = svList

    def restoreAfterRendering(self):
        self.stateVectors = self._cpStateVectors

    def initProperties(self,catalog):
        if 'state_vectors' in catalog:
            st = catalog['state_vectors']
            self.stateVectors = []
            for ls in st:
                self.stateVectors.append(StateVector(time=datetime.datetime.strptime(ls[0],'%Y-%m-%dT%H:%M:%S.%f'),pitch=ls[1],
                           roll=ls[2],yaw=ls[3]))

            catalog.pop('state_vectors')
        super().initProperties(catalog)

    @property
    def stateVectors(self):
        return self._stateVectors

    @stateVectors.setter
    def stateVectors(self,val):
        self._stateVectors = val
    ## A container needs a length.
    def __len__(self):
        return len(self.stateVectors)

    ## A container needs a getitem
    def __getitem__(self, index):
        return self.stateVectors[index]

    def __setitem__(self, *args):
        raise TypeError("'%s' object does not support item assignment" %
                        self.__class__.__name__
                        )

    def __delitem__(self, *args):
        raise TypeError("'%s' object does not support item deletion"
                        %self.__class__.__name__)

    def __iter__(self):
        return self

    def next(self):
        if self._last < len(self):
            result = self.stateVectors[self._last]
            self._last += 1
            return result
        raise StopIteration()

    def setAttitudeQuality(self, qual):
        self._attitudeQuality = qual

    def getAttitudeQuality(self):
        return self._attitudeQuality

    def setAttitudeSource(self, source):
        self._attitudeSource = source

    def getAttitudeSource(self):
        return self._attitudeSource

    @type_check(StateVector)
    def addStateVector(self, vec):
        self._stateVectors.append(vec)
        # Reset the minimum and maximum time bounds if necessary
        if (vec.time < self._minTime): self._minTime = vec.time
        if (vec.time > self._maxTime): self._maxTime = vec.time
        pass

    #TODO This needs to be fixed to work with scalar pitch, roll and yaw data
    #TODO- use Utils/geo/charts and let numpy do the work (JEB).
    def interpolate(self, time):
        if len(self) < self.min_length_for_interpolation:
            message = ("Fewer than %d state vectors present in attitude, "+
                       "cannot interpolate" % self.min_length_for_interpolation
                       )
            self.logger.error(
                message
                )
            return None
        if not self._inRange(time):
            message = (
                "Time stamp (%s) falls outside of the interpolation interval"+
                "[%s:%s]"
                ) % (time, self._minTime, self._maxTime)
            raise ValueError(message)
        pitch = 0.0
        roll = 0.0
        yaw = 0.0
        for sv1 in self.stateVectors:
            tmp=1.0
            for sv2 in self.stateVectors:
                if sv1.time == sv2.time:
                    continue
                numerator = float(self._timeDeltaToSeconds(sv2.time-time))
                denominator = float(
                    self._timeDeltaToSeconds(sv2.time - sv1.time)
                    )
                tmp *= numerator/denominator
                pass
            pitch += sv1.pitch*tmp
            roll  += sv1.roll*tmp
            yaw   += sv1.yaw*tmp
            pass
        return StateVector(name='asv', time=time, pitch=pitch, roll=roll, yaw=yaw)

    def _inRange(self, time):
        """Check whether a given time stamp is within the range of values for
        an orbit"""
        return self._minTime <= time <= self._maxTime

    @type_check(datetime.timedelta)
    def _timeDeltaToSeconds(self, td):
        return (
            td.microseconds +
            (td.seconds + td.days * 24.0 * 3600) * 10**6
            ) / 10**6

    def __str__(self):
        retstr = "Attitude Source: %s\n"
        retlst = (self.attitudeSource,)
        retstr += "Attitude Quality: %s\n"
        retlst += (self.attitudeQuality,)
        return retstr % retlst

    attitudeQuality = property(getAttitudeQuality, setAttitudeQuality)
    attitudeSource = property(getAttitudeSource, setAttitudeSource)
    pass


def createAttitude():
    return Attitude()
