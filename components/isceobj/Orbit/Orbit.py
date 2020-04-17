#
# Author: Walter Szeliga
# Copyright 2010, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged. Any commercial
# use must be negotiated with the Office of Technology Transfer at the
# California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before  exporting such information to
# foreign countries or providing access to foreign persons.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime
import numpy as np
import logging
import operator
from functools import reduce
#from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys import DateTimeUtil as DTU
from iscesys.Traits.Datetime import datetimeType
from iscesys.Component.Component import Component
from isceobj.Util.decorators import type_check, pickled, logged

# This class stores platform position and velocity information.
POSITION = Component.Parameter(
    '_position',
    public_name='POSITION',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    doc=''
)


TIME = Component.Parameter(
    '_time',
    public_name='TIME',
    default=[],
    type=datetimeType,
    mandatory=True,
    doc=''
)


VELOCITY = Component.Parameter(
    '_velocity',
    public_name='VELOCITY',
    default=[],
    container=list,
    type=float,
    mandatory=True,
    doc=''
)

class StateVector(Component):

    parameter_list = (POSITION,
                      TIME,
                      VELOCITY
                      )
    family = 'statevector'
    def __init__(self,family = None, name = None, time=None, position=None, velocity=None):
        super().__init__(
            family=family if family else  self.__class__.family, name=name)
        super(StateVector, self).__init__()
        self._time = time
        self._position = position or []
        self._velocity = velocity or []
        return None
    def __iter__(self):
        return self

    @type_check(datetime.datetime)
    def setTime(self, time):
        self._time = time
        pass

    def getTime(self):
        return self._time

    def setPosition(self, position):
        self._position = position

    def getPosition(self):
        return self._position

    def setVelocity(self, velocity):
        self._velocity = velocity

    def getVelocity(self):
        return self._velocity

    def getScalarVelocity(self):
        """Calculate the scalar velocity M{sqrt(vx^2 + vy^2 + vz^2)}.
        @rtype: float
        @return: the scalar velocity
        """
        return reduce(operator.add, [item**2 for item in self.velocity])**0.5

    def calculateHeight(self, ellipsoid):
        """Calculate the height above the provided ellipsoid.
        @type ellipsoid: Ellipsoid
        @param ellipsoid: an ellipsoid
        @rtype: float
        @return: the height above the ellipsoid
        """
        print("Orbit.calculateHeight: self.position = ", self.position)
        print("Orbit.calculateHeight: ellipsoid.a, ellipsoid.e2 = ",
            ellipsoid.a, ellipsoid.e2)
        lat, lon, height = ellipsoid.xyz_to_llh(self.position)
        return height

    def __lt__(self, other):
        return self.time < other.time
    def __gt__(self, other):
        return self.time > other.time
    def __cmp__(self, other):
        return  (self.time>other.time) - (self.time<other.time)
#    def __eq__(self, other):
#        return self._time == other._time
    def __eq__(self, other):
        return ((self._time == other._time) and
                (self._position == other._position) and
                (self._velocity == other._velocity))

    def __str__(self):
        retstr = "Time: %s\n"
        retlst = (self._time,)
        retstr += "Position: %s\n"
        retlst += (self._position,)
        retstr += "Velocity: %s\n"
        retlst += (self._velocity,)
        return retstr % retlst

    time = property(getTime,setTime)
    position = property(getPosition,setPosition)
    velocity = property(getVelocity,setVelocity)
    pass

STATE_VECTOR = Component.Facility('stateVector',
    public_name='STATE_VECTOR',
    module='isceobj.Orbit.Orbit',
    factory='StateVector',
    doc='State Vector properties used by STATE_VECTORS'
    )
STATE_VECTORS = Component.Facility('_stateVectors',
    public_name='STATE_VECTORS',
    module='iscesys.Component',
    factory='createTraitSeq',
    args=('statevector',),
    mandatory=False,
    doc='Testing Trait Sequence'
    )
ORBIT_QUALITY = Component.Parameter('_orbitQuality',
    public_name='ORBIT_QUALITY',
    default = '',
    type=str,
    mandatory=False,
    doc="Orbit quality"
    )
ORBIT_SOURCE = Component.Parameter('_orbitSource',
    public_name='ORBIT_SOURCE',
    default = '',
    type=str,
    mandatory=False,
    doc="Orbit source"
    )
ORBIT_REFERENCE_FRAME = Component.Parameter('_referenceFrame',
    public_name='ORBIT_REFERENCE_FRAME',
    default = '',
    type=str,
    mandatory=False,
    doc="Orbit reference frame"
    )
MIN_TIME = Component.Parameter('_minTime',
        public_name = 'MIN_TIME',
        default=datetime.datetime(year=datetime.MAXYEAR,month=12,day=31),
        type=datetimeType,
        mandatory=True,
        doc=''
)
MAX_TIME = Component.Parameter('_maxTime',
        public_name = 'MAX_TIME',
        default=datetime.datetime(year=datetime.MINYEAR,month=1,day=1),
        type=datetimeType,
        mandatory=True,
        doc=''
)


##
# This class encapsulates orbital information\n
# The Orbit class consists of a list of \c StateVector objects
# and provides an iterator over this list.
@pickled
class Orbit(Component):


    '''
    REMOVE completely once conversion to Configurable is done
    dictionaryOfVariables = {'STATE_VECTORS':
                                 ['_stateVectors',float, 'mandatory'],
                             'ORBIT_QUALITY': ['_orbitQuality',str,'optional'],
                             'ORBIT_SOURCE': ['_orbitSource',str, 'optional'],
                             'ORBIT_REFERENCE_FRAME':
                                 ['_referenceFrame',str,'optional']
                             }
    '''
    logging_name = "isce.Orbit"

#    _minTime = datetime.datetime(year=datetime.MAXYEAR,month=12,day=31)
#    _maxTime = datetime.datetime(year=datetime.MINYEAR,month=1,day=1)

    family = "orbit"
    parameter_list = (
                      ORBIT_QUALITY,
                      ORBIT_SOURCE,
                      ORBIT_REFERENCE_FRAME,
                      MIN_TIME,
                      MAX_TIME,
                      )
    facility_list = (STATE_VECTORS,)

    @logged
    def __init__(self,family = None, name = None, source=None, quality=None, stateVectors=None):
        super().__init__(family=family if family else  self.__class__.family,
                         name=name)
        self.configure()
        self._last = 0
        self._orbitQuality = quality or None
        self._orbitSource = source or None
        self._referenceFrame = None
        self._stateVectors.configure()
        #self._stateVectors = stateVectors or []
        self._cpStateVectors = []
        type(self._stateVectors)
        return None

    #since the dump works only with primitives, convert the self._stateVectors to a list of lists
    #instead of a list of state vectors
    #def dump(self,filename):
    #    self.adaptToRender()
    #    super(Orbit,self).dump(filename)
    #    #restore in original format
    #    self.restoreAfterRendering()

    #def load(self,filename):
    #    import copy
    #    import datetime
    #    # when loaded up the stateVectors are just list of lists and the starting time is
    #    # ins str format
    #    super(Orbit,self).load(filename)
    #    #make a copy
    #    cpStateVectors = copy.deepcopy(self._stateVectors)
    #    #convert the str into datetime
    #    cpStateVectors[3] = datetime.datetime.strptime(cpStateVectors[3],'%Y-%m-%dT%H:%M:%S.%f')
    #    #pack the orbit into stateVectors
    #    self._packOrbit(cpStateVectors[0], cpStateVectors[1], cpStateVectors[2], cpStateVectors[3])

    def adaptToRender(self):
        import copy
        # make a copy of the stateVectors to restore it after dumping
        self._cpStateVectors = copy.deepcopy(self._stateVectors)

        #self._stateVectors = list(self._unpackOrbit())
        #self._stateVectors[3] = self._stateVectors[3].strftime('%Y-%m-%dT%H:%M:%S.%f')

    def restoreAfterRendering(self):
        self._stateVectors = self._cpStateVectors

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.stateVectors)

    def __getitem__(self, index):
        return self.stateVectors[index]

    def __contains__(self, time):
        return self._inRange(time)

    @property
    def stateVectors(self):
        return self._stateVectors

    @property
    def maxTime(self):
        return self._maxTime

    @maxTime.setter
    def maxTime(self, time):
        self._maxTime = time

    @property
    def minTime(self):
        return self._minTime

    @minTime.setter
    def minTime(self, time):
        self._minTime = time


    def setOrbitQuality(self,qual):
        self._orbitQuality = qual

    def getOrbitQuality(self):
        return self._orbitQuality

    def setOrbitSource(self,source):
        self._orbitSource = source

    def getOrbitSource(self):
        return self._orbitSource

    def setReferenceFrame(self,ref):
        self._referenceFrame = ref

    def getReferenceFrame(self):
        return self._referenceFrame

    @type_check(StateVector)
    def addStateVector(self, vec):
        """
        Add a state vector to the orbit.
        @type vec: Orbit.StateVector
        @param vec: a state vector
        @raise TypeError: if vec is not of type StateVector
        """
        pos = vec.getPosition()
        import math
        posmag = math.sqrt(sum([x*x for x in pos]))
        if posmag > 1.e10:
            print("Orbit.addStateVector: vec = ", vec)
            import sys
            sys.exit(0)

        vtime  = vec.getTime()
        if vtime > self.maxTime:
            self._stateVectors.append(vec)
        else:
            for ind, sv in enumerate(self._stateVectors):
                if sv.time > vtime:
                    break

            self._stateVectors.insert(ind, vec)

        # Reset the minimum and maximum time bounds if necessary
        if vec.time < self.minTime: self.minTime = vec._time
        if vec.time > self.maxTime: self.maxTime = vec._time

    def __next__(self):
        if self._last < len(self):
            next = self._stateVectors[self._last]
            self._last += 1
            return next
        else:
            self._last = 0 # This is so that we can restart iteration
            raise StopIteration()


    def interpolateOrbit(self, time, method='linear'):
        """Interpolate the state vector of the orbit at a given time.
        @type time: datetime.datetime
        @param time: the time at which interpolation is desired
        @type method: string
        @param method: the interpolation method, valid values are 'linear',
        'legendre' and 'hermite'
        @rtype: Orbit.StateVector
        @return: a state vector at the desired time otherwise None
        @raises ValueError: if the time lies outside of the time spanned by
        the orbit
        @raises NotImplementedError: if the desired interpolation method
        cannot be decoded
        """
        if time not in self:
            raise ValueError(
                "Time stamp (%s) falls outside of the interpolation interval [%s:%s]" %
                (time,self.minTime,self.maxTime)
                )

        if method == 'linear':
            newSV = self._linearOrbitInterpolation(time)
        elif method == 'legendre':
            newSV = self._legendreOrbitInterpolation(time)
        elif method == 'hermite':
            newSV = self._hermiteOrbitInterpolation(time)
        else:
            raise NotImplementedError(
                "Orbit interpolation type %s, is not implemented" % method
                )
        return newSV

    ## Isn't orbit redundant? -compute the method based on name
    def interpolate(self, time, method='linear'):
        if time not in self:
            raise ValueError("Time stamp (%s) falls outside of the interpolation interval [%s:%s]"
                             % (time,self.minTime,self.maxTime))
        try:
            return getattr(self, '_'+method+'OrbitInterpolation')(time)
        except AttributeError:
            pass
        raise NotImplementedError(
            "Orbit interpolation type %s, is not implemented" % method
            )

    interpolateOrbit = interpolate

    def _linearOrbitInterpolation(self,time):
        """
        Linearly interpolate a state vector.  This method returns None if
        there are fewer than 2 state vectors in the orbit.
        @type time: datetime.datetime
        @param time: the time at which to interpolate a state vector
        @rtype: Orbit.StateVector
        @return: the state vector at the desired time
        """
        if len(self) < 2:
            self.logger.error("Fewer than 2 state vectors present in orbit, cannot interpolate")
            return None

        position = [0.0 for i in range(3)]
        velocity = [0.0 for i in range(3)]
        newOrbit = self.selectStateVectors(time, 1, 1)
        obsTime, obsPos, obsVel, offset = newOrbit.to_tuple(
            relativeTo=self.minTime
        )
        dtime = float(DTU.timeDeltaToSeconds(time-offset))

        for i in range(3):
            position[i] = (obsPos[0][i] +
                (obsPos[1][i]-obsPos[0][i])*
                (dtime-obsTime[0])/(obsTime[1]-obsTime[0]))
            velocity[i] = (obsVel[0][i] +
                (obsVel[1][i]-obsVel[0][i])*
                (dtime-obsTime[0])/(obsTime[1]-obsTime[0]))

        """
        for sv1 in self.stateVectors:
            tmp=1.0
            for sv2 in self.stateVectors:
                if sv1.time == sv2.time:
                    continue
                numerator = float(DTU.timeDeltaToSeconds(sv2.time-time))
                denominator = float(
                    DTU.timeDeltaToSeconds(sv2.time - sv1.time)
                    )
                tmp = tmp*(numerator)/(denominator)
            for i in range(3):
                position[i] = position[i] + sv1.getPosition()[i]*tmp
                velocity[i] = velocity[i] + sv1.getVelocity()[i]*tmp
        """
        return StateVector(time=time, position=position, velocity=velocity)

    def _legendreOrbitInterpolation(self,time):
        """Interpolate a state vector using an 8th order Legendre polynomial.
        This method returns None if there are fewer than 9 state vectors in
        the orbit.
        @type time: datetime.datetime
        @param time: the time at which to interpolate a state vector
        @rtype: Orbit.StateVector
        @return: the state vector at the desired time
        """
        if len(self) < 9:
            self.logger.error(
                "Fewer than 9 state vectors present in orbit, cannot interpolate"
                )
            return None

        seq = [4,5,3,6,2,7,1,8]
        found = False

        for ind in seq:
            rind = 9 - ind
            try:
                newOrbit = self.selectStateVectors(time, 4, 5)
                found = True
            except LookupError as e:
                pass

            if found:
                break

        if not found:
            raise Exception('Could not find state vectors before/after for interpolation')


        obsTime, obsPos, obsVel, offset = newOrbit.to_tuple(
        relativeTo=self.minTime
        )
        t = DTU.timeDeltaToSeconds(time-self.minTime)
        t0 = DTU.timeDeltaToSeconds(newOrbit.minTime-self.minTime)
        tn = DTU.timeDeltaToSeconds(newOrbit.maxTime-self.minTime)
        ansPos = self._legendre8(t0, tn, t, obsPos)
        ansVel = self._legendre8(t0, tn, t, obsVel)

        return StateVector(time=time, position=ansPos, velocity=ansVel)



    def _legendre8(self,t0,tn,t,v):
        """Interpolate an orbit using an 8th order Legendre polynomial
        @type t0: float
        @param t0: starting time
        @type tn: float
        @param tn: ending time
        @type t: float
        @param t: time at which vt must be interpolated
        @type v: list
        @param v: 9 consecutive points
        @rtype: float
        @return: interpolated point at time t
        """
        trel = (t-t0)/(tn-t0)*(len(v)-1)+1
        itrel=max(1,min(int(trel)-4,len(v)-9))+1
        t = trel-itrel
        vt = [0 for i in range(3)]
        kx = 0
        x=t+1
        noemer = [40320,-5040,1440,-720,576,-720,1440,-5040,40320]

        teller=(x)*(x-1)*(x-2)*(x-3)*(x-4)*(x-5)*(x-6)*(x-7)*(x-8)
        if (teller == 0):
            kx = int(x)
            for i in range(3):
                vt[i] = v[kx][i]
        else:
            for kx in range(9):
                coeff=teller/noemer[kx]/(x-kx)
                for i in range(3):
                    vt[i] = vt[i] + coeff*v[kx][i]

        return vt


    def _hermiteOrbitInterpolation(self,time):
        """
        Interpolate a state vector using Hermite interpolation.
        This method returns None if there are fewer than 4 state
        vectors in the orbit.
        @type time: datetime.datetime
        @param time: the time at which to interpolate a state vector
        @rtype: Orbit.StateVector
        @return: the state vector at the desired time
        """

        import os
        from ctypes import c_double, cdll,byref
        orbitHermite = (
            cdll.LoadLibrary(os.path.dirname(__file__)+'/orbitHermite.so')
            )

        if len(self) < 4:
            self.logger.error(
                "Fewer than 4 state vectors present in orbit, cannot interpolate"
                )
            return None

        # The Fortran routine assumes that it is getting an array of four
        # state vectors
        try:
            newOrbit = self.selectStateVectors(time, 2, 2)
        except LookupError:
            try:
                newOrbit = self.selectStateVectors(time,1,3)
            except LookupError:
                try:
                    newOrbit = self.selectStateVectors(time,3,1)
                except LookupError:
                    self.logger.error("Unable to select 2 state vectors before and after "+
                                       "chosen time %s" % (time))
                    return None

        # For now, assume that time is an array containing the times at which
        # we want to interpolate
        obsTime, obsPos, obsVel,offset = newOrbit.to_tuple(
            relativeTo=self.minTime
            )

        td = time - self.minTime
        ansTime = DTU.timeDeltaToSeconds(td)
        flatObsPos = [item for sublist in obsPos for item in sublist]
        flatObsVel = [item for sublist in obsVel for item in sublist]
        flatAnsPos= [0.,0.,0.]# list([0.0 for i in range(3)])
        flatAnsVel= [0.,0.,0.]#list([0.0 for i in range(3)])
        obsTime_C = (c_double*len(obsTime))(*obsTime)
        obsPos_C = (c_double*len(flatObsPos))(*flatObsPos)
        obsVel_C = (c_double*len(flatObsVel))(*flatObsVel)
        ansTime_C = c_double(ansTime)
        ansPos_C = (c_double*3)(*flatAnsPos)
        ansVel_C = (c_double*3)(*flatAnsVel)

        # Use the C wrapper to the fortran Hermite interpolator
        orbitHermite.orbitHermite_C(obsPos_C,
                                    obsVel_C,
                                    obsTime_C,
                                    byref(ansTime_C),
                                    ansPos_C,
                                    ansVel_C)

        return StateVector(time=time, position=ansPos_C[:], velocity=ansVel_C[:])

    ## This need to be public -very confusing since there is an __iter__
    def to_tuple(self, relativeTo=None):
        return self._unpackOrbit(relativeTo=relativeTo)

    def _unpackOrbit(self, relativeTo=None):
        """Convert and orbit object into tuple of lists containing time,
        position and velocity.
        @type relativeTo: datetime.datetime
        @param relativeTo: the time with which to reference the unpacked orbit
        @return: a tuple containing a list of time, position, velocity and
        relative time offset
        """
        time = []
        position = []
        velocity = []
        if relativeTo is None:
            relativeTo = self.minTime

        for sv in self.stateVectors:
            td = sv.time - relativeTo
            currentTime = ((
                    td.microseconds +
                    (td.seconds + td.days * 24 * 3600.0) * 10**6) / 10**6
                           )
            currentPosition = sv.getPosition()
            currentVelocity = sv.getVelocity()
            time.append(currentTime)
            position.append(currentPosition)
            velocity.append(currentVelocity)

        return time, position, velocity, relativeTo

    #def _packOrbit(self,time,position,velocity,relativeTo):
    #    self._minTime = relativeTo
    #    self._stateVectors = [];
    #    for t,p,v in zip(time,position,velocity):
    #        sv = StateVector(time=relativeTo + datetime.timedelta(seconds=t),position=p,velocity=v)
    #        self.addStateVector(sv)

    ## Why does this version fail ERS and not ALOS?
    def selectStateVectorsBroken(self, time, before, after):
        """Given a time and a number of before and after state vectors,
        return an Orbit with (before+after) state vectors with reference to
        time.
        @type time: datetime.datetime
        @param time: the reference time for subselection
        @type before: integer
        @param before: the number of state vectors before the chosen time to
        select
        @type after: integer
        @param after: the number of state vectors after the chosen time to
        select
        @rtype: Orbit.Orbit
        @return: an orbit containing (before+after) state vectors relative to
        time
        @raises LookupError: if there are insufficient state vectors in the
        orbit
        """
        # First, find the index closest to the requested time
        i=0
        while self.stateVectors[i].time <= time:
            i += 1
        beforeIndex = i

        # Check that we can grab enough data
        if (beforeIndex-before) < 0:
            raise LookupError("Requested index %s is out of bounds" %
                              (beforeIndex-before))
        elif (beforeIndex+after) > len(self):
            raise LookupError("Requested index %s is out of bounds" %
                              (beforeIndex+after))

        # Create a new orbit object - filled with goodies.
        return Orbit(source=self.orbitSource,
                     quality=self.orbitQuality,
                     stateVectors=[
                self[i] for i in range(
                    (beforeIndex-before),(beforeIndex+after)
                    )])



    def selectStateVectors(self,time,before,after):
        """
        Given a time and a number of before and after state vectors,
        return an Orbit with (before+after) state vectors with reference to
        time.
        """
        # First, find the index closest to the requested time
        i=0
        while(self._stateVectors[i].getTime() <= time):
            i += 1
        beforeIndex = i

        # Check that we can grab enough data
        if ((beforeIndex-before) < 0):
            raise LookupError(
                "Requested index %s is out of bounds" % (beforeIndex-before)
                )
        elif ((beforeIndex+after) > len(self._stateVectors)):
            raise LookupError(
                "Requested index %s is out of bounds" % (beforeIndex+after)
                )

        # Create a new orbit object
        newOrbit = Orbit(name='neworbit')
        newOrbit.configure()
        # inject dependencies
        newOrbit.setOrbitSource(self.orbitSource)
        newOrbit.setOrbitQuality(self.orbitQuality)
        for i in range((beforeIndex-before),(beforeIndex+after)):
            newOrbit.addStateVector(self[i])

        return newOrbit



    def trimOrbit(self, startTime, stopTime):
        """Trim the list of state vectors to encompass the time span
        [startTime:stopTime]
        @type startTime: datetime.datetime
        @param startTime: the desired starting time for the output orbit
        @type stopTime: datetime.datetime
        @param stopTime: the desired stopping time for the output orbit
        @rtype: Orbit.Orbit
        @return: an orbit containing all of the state vectors within the time
        span [startTime:stopTime]
        """

        newOrbit = Orbit()
        newOrbit.configure()
        newOrbit.setOrbitSource(self._orbitSource)
        newOrbit.setReferenceFrame(self._referenceFrame)
        for sv in self._stateVectors:
            if startTime < sv.time < stopTime:
                newOrbit.addStateVector(sv)

        return newOrbit

    def _inRange(self,time):
        """Check whether a given time is within the range of values for an
        orbit.
        @type time: datetime.datetime
        @param time: a time
        @rtype: boolean
        @return: True if the time falls within the time span of the orbit,
        otherwise False
        """
        return self.minTime <= time <= self.maxTime

    def __str__(self):
        retstr = "Orbit Source: %s\n"
        retlst = (self._orbitSource,)
        retstr += "Orbit Quality: %s\n"
        retlst += (self._orbitQuality,)
        retstr += "Orbit Reference Frame: %s\n"
        retlst += (self._referenceFrame,)
        return retstr % retlst

    stateVector = property()
    orbitQuality = property(getOrbitQuality, setOrbitQuality)
    orbitSource = property(getOrbitSource, setOrbitSource)

    pass


    def getHeading(self, time=None, spacing=0.5, planet=None):
        '''
        Compute heading around given azimuth time.
        If time is not provided, mid point of orbit is used.
        '''

        from isceobj.Planet.Planet import Planet

        if planet is None:
            planet = Planet(pname='Earth')

        refElp = planet.ellipsoid
        if time is None:
            delta = self.maxTime - self.minTime
            aztime = self.minTime + datetime.timedelta(seconds = 0.5 * delta.total_seconds())
        else:
            aztime = time

        t1 = aztime - datetime.timedelta(seconds=spacing)
        t2 = aztime + datetime.timedelta(seconds=spacing)

        vec1 = self.interpolateOrbit(t1, method='hermite')
        vec2 = self.interpolateOrbit(t2, method='hermite')

        llh1 = refElp.xyz_to_llh(vec1.getPosition())
        llh2 = refElp.xyz_to_llh(vec2.getPosition())

        #Heading
        hdg = refElp.geo_hdg(llh1, llh2)

        return np.degrees(hdg)

    def getENUHeading(self, time=None, planet=None):
        '''
        Compute heading at given azimuth time using single state vector.
        If time is not provided, mid point of orbit is used.
        '''

        from isceobj.Planet.Planet import Planet

        if planet is None:
            planet = Planet(pname='Earth')

        refElp = planet.ellipsoid
        if time is None:
            delta = self.maxTime - self.minTime
            aztime = self.minTime + datetime.timedelta(seconds = 0.5 * delta.total_seconds())
        else:
            aztime = time

        vec1 = self.interpolateOrbit(aztime, method='hermite')
        llh1 = refElp.xyz_to_llh(vec1.getPosition())

        enumat = refElp.enubasis(llh1)
        venu = np.dot(enumat.xyz_to_enu, vec1.getVelocity())

        #Heading
        hdg = np.arctan2(venu[0,0], venu[0,1])

        return np.degrees(hdg)


    def rdr2geo(self, aztime, rng, height=0.,
            doppler = None, wvl = None,
            planet=None, side=-1):
        '''
        Returns point on ground at given height and doppler frequency.
        Never to be used for heavy duty computing.
        '''

        from isceobj.Planet.Planet import Planet

        ####Setup doppler for the equations
        dopfact = 0.0

        hdg = self.getENUHeading(time=aztime)

        sv = self.interpolateOrbit(aztime, method='hermite')
        pos = sv.getPosition()
        vel = sv.getVelocity()
        vmag = np.linalg.norm(vel)

        if doppler is not None:
            dopfact = doppler(DTU.seconds_since_midnight(aztime), rng) * 0.5 * wvl * rng/vmag

        if planet is None:
            refElp = Planet(pname='Earth').ellipsoid
        else:
            refElp = planet.ellipsoid

        ###Convert position and velocity to local tangent plane
        satLLH = refElp.xyz_to_llh(pos)

        refElp.setSCH(satLLH[0], satLLH[1], hdg)
        radius = refElp.pegRadCur

        #####Setup ortho normal system right below satellite
        satVec = np.array(pos)
        velVec = np.array(vel)

        ###Setup TCN basis
        clat = np.cos(np.radians(satLLH[0]))
        slat = np.sin(np.radians(satLLH[0]))
        clon = np.cos(np.radians(satLLH[1]))
        slon = np.sin(np.radians(satLLH[1]))
        nhat = np.array([-clat*clon, -clat*slon, -slat])
        temp = np.cross(nhat, velVec)
        chat = temp / np.linalg.norm(temp)
        temp = np.cross(chat, nhat)
        that = temp / np.linalg.norm(temp)
        vhat = velVec / np.linalg.norm(velVec)

        ####Solve the range doppler eqns iteratively
        ####Initial guess
        zsch = height

        for ii in range(10):

            ###Near nadir tests
            if (satLLH[2]-zsch) >= rng:
                return None 

            a = (satLLH[2] + radius)
            b = (radius + zsch)

            costheta = 0.5*(a/rng + rng/a - (b/a)*(b/rng))
            sintheta = np.sqrt(1-costheta*costheta)

            gamma = rng*costheta
            alpha = dopfact - gamma*np.dot(nhat,vhat)/np.dot(vhat,that)
            beta = -side*np.sqrt(rng*rng*sintheta*sintheta - alpha*alpha)

            delta = alpha * that + beta * chat + gamma * nhat

            targVec = satVec + delta

            targLLH = refElp.xyz_to_llh(list(targVec))
            targXYZ = refElp.llh_to_xyz([targLLH[0], targLLH[1], height])
            targSCH = refElp.xyz_to_sch(targXYZ)

            zsch = targSCH[2]

            rdiff  = rng - np.linalg.norm(np.array(satVec) - np.array(targXYZ))

        return targLLH


    def rdr2geoNew(self, aztime, rng, height=0.,
            doppler = None, wvl = None,
            planet=None, side=-1):
        '''
        Returns point on ground at given height and doppler frequency.
        Never to be used for heavy duty computing.
        '''

        from isceobj.Planet.Planet import Planet

        ####Setup doppler for the equations
        dopfact = 0.

        sv = self.interpolateOrbit(aztime, method='hermite')
        pos = np.array(sv.getPosition())
        vel =np.array( sv.getVelocity())
        vmag = np.linalg.norm(vel)

        if doppler is not None:
            dopfact = doppler(DTU.seconds_since_midnight(aztime), rng) * 0.5 * wvl * rng/vmag

        if planet is None:
            refElp = Planet(pname='Earth').ellipsoid
        else:
            refElp = planet.ellipsoid

        ###Convert position and velocity to local tangent plane
        major = refElp.a
        minor = major * np.sqrt(1 - refElp.e2)

        #####Setup ortho normal system right below satellite
        satDist = np.linalg.norm(pos)
        alpha = 1 / np.linalg.norm(pos/ np.array([major, major, minor]))
        radius = alpha * satDist
        hgt = (1.0 - alpha) * satDist

        ###Setup TCN basis - Geocentric
        nhat = -pos/satDist
        temp = np.cross(nhat, vel)
        chat = temp / np.linalg.norm(temp)
        temp = np.cross(chat, nhat)
        that = temp / np.linalg.norm(temp)
        vhat = vel / vmag

        ####Solve the range doppler eqns iteratively
        ####Initial guess
        zsch = height

        for ii in range(10):

            ###Near nadir tests
            if (hgt-zsch) >= rng:
                return None 

            a = satDist
            b = (radius + zsch)

            costheta = 0.5*(a/rng + rng/a - (b/a)*(b/rng))
            sintheta = np.sqrt(1-costheta*costheta)

            gamma = rng*costheta
            alpha = dopfact - gamma*np.dot(nhat,vhat)/np.dot(vhat,that)
            beta = -side*np.sqrt(rng*rng*sintheta*sintheta - alpha*alpha)

            delta = alpha * that + beta * chat + gamma * nhat

            targVec = pos + delta

            targLLH = refElp.xyz_to_llh(list(targVec))
            targXYZ = np.array(refElp.llh_to_xyz([targLLH[0], targLLH[1], height]))

            zsch = np.linalg.norm(targXYZ) - radius

            rdiff  = rng - np.linalg.norm(pos - targXYZ)

        return targLLH


    ####Make rdr2geo same as pointOnGround
    pointOnGround = rdr2geo

    def geo2rdr(self, llh, side=-1, planet=None,
            doppler=None, wvl=None):
        '''
        Takes a lat, lon, height triplet and returns azimuth time and range.
        Assumes zero doppler for now.
        '''

        from isceobj.Planet.Planet import Planet
        from isceobj.Util.Poly2D import Poly2D
        if doppler is None:
            doppler = Poly2D()
            doppler.initPoly(azimuthOrder=0, rangeOrder=0, coeffs=[[0.]])
            wvl = 0.0

        if planet is None:
            refElp = Planet(pname='Earth'). ellipsoid
        else:
            refElp = planet.ellipsoid

        xyz = refElp.llh_to_xyz(llh)

        delta = (self.maxTime - self.minTime).total_seconds() * 0.5
        tguess = self.minTime + datetime.timedelta(seconds = delta)
        outOfBounds = False
        for ii in range(51):
            try:
                sv = self.interpolateOrbit(tguess, method='hermite')
            except:
                outOfBounds = True
                break

            pos = np.array(sv.getPosition())
            vel = np.array(sv.getVelocity())

            dr = xyz-pos
            rng = np.linalg.norm(dr)

            dopfact = np.dot(dr,vel)
            fdop = doppler(DTU.seconds_since_midnight(tguess),rng) * wvl * 0.5
            fdopder = (0.5*wvl*doppler(DTU.seconds_since_midnight(tguess),rng+10.0) - fdop) / 10.0

            fn = dopfact - fdop * rng
            c1 = -np.dot(vel, vel)
            c2 = (fdop/rng + fdopder)

            fnprime = c1 + c2 * dopfact

            tguess = tguess - datetime.timedelta(seconds = fn/fnprime)

        if outOfBounds:
            raise Exception('Interpolation time out of bounds')


        return tguess, rng


    def exportToC(self, reference=None):
        from isceobj.Util import combinedlibmodule
        orb = []

        ###Continue usage as usual if no reference is provided
        ###This wont break the old interface but could cause 
        ###issues at midnight crossing
        if reference is None:
            reference = self.minTime

        refEpoch = reference.replace(hour=0, minute=0, second=0, microsecond=0)

        for sv in self._stateVectors:
            tim = (sv.getTime() - refEpoch).total_seconds()
            pos = sv.getPosition()
            vel = sv.getVelocity()

            row = [tim] + pos + vel
            orb.append(row)

        cOrbit = combinedlibmodule.exportOrbitToC(1,orb)
        return cOrbit

    def importFromC(self, ptr, dateobj):
        from isceobj.Util import combinedlibmodule
        from datetime import timedelta

        print('Importing from C')
        basis, data = combinedlibmodule.importOrbitFromC(ptr)

        for row in data:
            sv = StateVector()
            sv.setTime( dateobj + timedelta(seconds = row[0]))
            sv.setPosition(row[1:4])
            sv.setVelocity(row[4:7])
            self.addStateVector(sv)

        return


def createOrbit():
    return Orbit()
