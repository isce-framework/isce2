#!/usr/bin/env python3
import stdproc
import datetime
from .Orbit import StateVector
from isceobj.Util.geo.ellipsoid import Ellipsoid
from iscesys.StdOEL.StdOELPy import create_writer
from isceobj.Location.Peg import Peg 
from iscesys.Component.Component import Component
from isceobj.Planet.Planet import Planet
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
import numpy as np 


NEW_POINTS = Component.Parameter('_newPoints',
        public_name='NEW_POINTS',
        default = 2,
        type = int,
        mandatory=False,
        doc = 'Number of points to add the start and end of current orbit')

POLYNOMIAL_ORDER = Component.Parameter('_polyOrder',
        public_name='POLYNOMIAL_ORDER',
        default=2,
        type=int,
        mandatory=False,
        doc='Order of the polynomial to use for SCH interpolation')


def diffOrbits(orig, new, skip=2):
    '''
    Compute statistics between old and new orbits.
    Points in the middle were transformed from WGS84 -> SCH -> WGS84.
    This reports the error in the transformation.
    '''

    oldnum = len(orig._stateVectors)
    res = np.zeros((oldnum, 6))

    for kk,sv in enumerate(orig):
        newsv = new[kk+skip]
        res[kk,0:3] = np.array(sv.getPosition()) - np.array(newsv.getPosition())
        res[kk,3:6] = np.array(sv.getVelocity()) - np.array(newsv.getVelocity())

    print('RMS error from interpolation: ')
    print(np.sqrt(np.mean(res*res, axis=0)))

class OrbitExtender(Component):
    '''
    Code to extrapolate WGS84 orbits by a few points. Orbit is transformed in to a SCH coordinate system and transferred back to WGS84.'''

    family = 'orbitextender'
    logging_name='isceobj.orbitextender'
    _planet = None
    parameter_list = (NEW_POINTS,
                      POLYNOMIAL_ORDER)


    def __init__(self, name='', num=None, order=None, planet=None):
        super(OrbitExtender,self).__init__(family=self.__class__.family, name=name)
        if num is not None:
            self._newPoints = int(num)

        if order is not None:
            self._polyOrder = int(order)

        if planet is not None:
            self._planet = planet
        else:
            self._planet = Planet(pname='Earth')

    def getPegAndHeading(self, orbit, midTime, delta=5000):
        '''Compute the heading of the satellite and peg lat, lon'''

        
        refElp = Ellipsoid(a=self._planet.ellipsoid.a, e2=self._planet.ellipsoid.e2, model='WGS84')

        #Position just before mid Time
        t1 = midTime - datetime.timedelta(microseconds=delta) 
        vec1 = orbit.interpolate(t1, method='hermite')

        #Position just after midTime
        t2 = midTime + datetime.timedelta(microseconds=delta)
        vec2 = orbit.interpolate(t2, method='hermite')

        pos = vec1.getPosition()
        pt1 = refElp.ECEF(pos[0], pos[1], pos[2])
        pos = vec2.getPosition()
        pt2 = refElp.ECEF(pos[0], pos[1], pos[2])

        llh1 = pt1.llh()
        llh2 = pt2.llh()

        #Heading
        hdg = pt1.bearing(pt2)

        #Average lat lon
        peg = refElp.LLH(0.5*(llh1.lat + llh2.lat), 0.5*(llh1.lon + llh2.lon), 0.5*(llh1.hgt+llh2.hgt))
        return peg, hdg

    def getSCHOrbit(self, orbit, peg, hdg):
        '''
        Accepts a WGS-84 orbit and converts it to SCH.
        '''
        writer = create_writer("log","",True,filename='orbit_extender.log')
        llh = [peg.lat, peg.lon, peg.hgt]

        ####Local radius
        radius = self._planet.ellipsoid.radiusOfCurvature(llh, hdg=hdg)

        #midPeg is a Location.Peg object
        midPeg = Peg(latitude = peg.lat,
                         longitude = peg.lon,
                         heading = hdg,
                         radiusOfCurvature = radius)

            
        orbSch = stdproc.createOrbit2sch(averageHeight = peg.hgt)
        orbSch.setStdWriter(writer)
        orbSch(planet=self._planet, orbit=orbit, peg=midPeg)

        return orbSch.orbit
           
    def extendSCHOrbit(self, orbit):
        '''
        Extends a given SCH orbit by _newPoints and using a 
        polynomial of order _polyOrder.
        '''

        lenv = len(orbit)

        t = np.zeros(lenv)
        pos = np.zeros((lenv,6))

        t0 = orbit[0].getTime()

        ####Read in data in to numpy arrays
        for kk,sv in enumerate(orbit):
            t[kk] = float((sv.getTime()-t0).total_seconds())
            pos[kk,0:3] = sv.getPosition()
            pos[kk,3:6] = sv.getVelocity()

        ####Interpolation at top of the array
        delta = t[1] - t[0]
        ttop = delta*np.arange(-self._newPoints,0)
        toppos = np.zeros((self._newPoints,6))

        x = t[0:self._polyOrder+1]
        y = pos[0:self._polyOrder+1,:]

        ###Simple polynomial interpolation for each coordinate
        for kk in range(6):
            toppoly = np.polyfit(x,y[:,kk],self._polyOrder)
            toppos[:,kk] = np.polyval(toppoly, ttop)

        for kk in range(self._newPoints):
            sv = StateVector()
            sv.setTime(t0 + datetime.timedelta(seconds=ttop[kk]))
            sv.setPosition(list(toppos[kk,0:3]))
            sv.setVelocity(list(toppos[kk,3:6]))
            orbit._stateVectors.insert(kk,sv)

        orbit._minTime = orbit[0].getTime()


        ###Interpolate at the bottom
        delta = t[-1] - t[-2]
        tbot = t[-1] + delta* np.arange(1, self._newPoints+1)
        botpos = np.zeros((self._newPoints,6)) 

        x = t[-self._polyOrder-1:]
        y = pos[-self._polyOrder-1:,:]
        for kk in range(6):
            botpoly = np.polyfit(x,y[:,kk],self._polyOrder)
            botpos[:,kk] = np.polyval(botpoly,tbot)

        for kk in range(self._newPoints):
            sv = StateVector()
            sv.setTime(t0 + datetime.timedelta(seconds=tbot[kk]))
            sv.setPosition(list(botpos[kk,0:3]))
            sv.setVelocity(list(botpos[kk,3:6]))
            orbit._stateVectors.append(sv)

        orbit._maxTime = orbit[-1].getTime()

        return

    def getXYZOrbit(self, orbit, peg, hdg):
        '''
        Convert an input SCH orbit to XYZ coords.
        '''
        llh = [peg.lat, peg.lon, peg.hgt]
        radius = self._planet.ellipsoid.radiusOfCurvature(llh, hdg=hdg)

        midPeg = Peg(latitude=peg.lat,
                     longitude=peg.lon,
                     heading=hdg,
                     radiusOfCurvature=radius)
        writer = create_writer("log","",True,filename='orbit_extender.log')
        orbxyz = stdproc.createSch2orbit()
        orbxyz.radiusOfCurvature = radius
        orbxyz.setStdWriter(writer)
        orbxyz(planet=self._planet, orbit=orbit, peg=midPeg)
        return orbxyz.orbit

        
    def extendOrbit(self, orbit):
        '''
        Input orbit must be WGS-84.
        '''

        deltaT = DTUtil.timeDeltaToSeconds(orbit.maxTime - orbit.minTime)/2.0
        midTime = orbit.minTime + datetime.timedelta(microseconds=int(deltaT*1e6))

        #pegCoord is an Util.geo coordinate object
        pegCoord, hdg = self.getPegAndHeading(orbit, midTime)

        ####Sch orbit w.r.t mid point of orbit
        schOrb = self.getSCHOrbit(orbit, pegCoord, hdg)

        ####Extend the SCH orbits
        self.extendSCHOrbit(schOrb)

        ####Convert the SCH orbit back to WGS84 orbits
        extOrb = self.getXYZOrbit(schOrb, pegCoord, hdg)

        ####Statistics on the transforms if needed
        #diffOrbits(orbit, extOrb, skip=self._newPoints)

        return extOrb 
