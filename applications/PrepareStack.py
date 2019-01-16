#!/usr/bin/env python3
from __future__ import print_function
import argparse
import isce
from make_raw import makeRawApp
import numpy as np
import os
import itertools
from isceobj.XmlUtil.XmlUtil import XmlUtil
from isceobj.Orbit.Orbit import Orbit, StateVector
from iscesys.StdOEL.StdOELPy import create_writer
#import sarxml
import stdproc
import datetime



stdWriter = create_writer("log", "", True, filename="prepareStack.log")

def pulseTiming(frame):
    #From runPulseTiming() in InsarProc
    numberOfLines = frame.getNumberOfLines()
    prf = frame.getInstrument().getPulseRepetitionFrequency()
    pri = 1.0 / prf
    startTime = frame.getSensingStart()
    orbit = frame.getOrbit()

    pulseOrbit = Orbit()
    startTimeUTC0 = (startTime - datetime.datetime(startTime.year,startTime.month,startTime.day))
    timeVec = [pri*i + startTimeUTC0.seconds + 10**-6*startTimeUTC0.microseconds for i in xrange(numberOfLines)]
    for i in range(numberOfLines):
        dt = i * pri
        time = startTime + datetime.timedelta(seconds=dt)
        sv = orbit.interpolateOrbit(time, method='hermite')
        pulseOrbit.addStateVector(sv)

    return pulseOrbit

def getPeg(planet, orbit):
    #Returns relevant peg point. From runSetMocompPath.py

    objPeg = stdproc.createGetpeg()
    objPeg.wireInputPort(name='planet', object=planet)
    objPeg.wireInputPort(name='Orbit', object=orbit)

    stdWriter.setFileTag("getpeg", "log")
    stdWriter.setFileTag("getpeg", "err")
    stdWriter.setFileTag("getpeg", "out")
#    objSetmocomppath.setStdWriter(self._stdWriter)
    objPeg.setStdWriter(stdWriter)
    objPeg.estimatePeg()

    return objPeg.getPeg(), objPeg.getAverageHeight()

class orbit_info:
    def __init__(self, sar, fname):
        '''Initialize with a sarProc object and corresponding XML file name'''
        orbit = pulseTiming(sar.make_raw.frame)
        tim, pos, vel, offset = orbit._unpackOrbit()
        planet = sar.make_raw.planet
        self.tim = tim
        self.pos = pos
        self.vel = vel
        self.dt = sar.make_raw.frame.sensingMid
        self.prf = sar.make_raw.doppler.prf
        self.fd  = sar.make_raw.dopplerValues() * self.prf
        self.nvec = len(self.tim)
        self.peg, self.hgt  = getPeg(planet, orbit)
        self.rds = self.peg.getRadiusOfCurvature()
        self.rng = sar.make_raw.frame.startingRange
        self.clook = None
        self.slook = None
        self.filename = fname
        self.computeLookAngle()

    def computeLookAngle(self):
        self.clook = (2*self.hgt*self.rds+self.hgt**2+self.rng**2)/(2*self.rng*(self.rds+self.hgt))
        self.slook = np.sqrt(1-self.clook**2)
#        print('Estimated Look Angle: %3.2f degrees'%(np.arccos(self.clook)*180.0/np.pi))

    def getBaseline(self, slave):
        '''Compute baseline between current object and another orbit object.'''

        ind = np.int(self.nvec/2)

        mpos = np.array(self.pos[ind])
        mvel = np.array(self.vel[ind])

        #######From the ROI-PAC scripts
        rvec = mpos/np.linalg.norm(mpos)
        crp = np.cross(rvec, mvel)/np.linalg.norm(mvel)
        crp = crp/np.linalg.norm(crp)
        vvec = np.cross(crp, rvec)
        mvel = np.linalg.norm(mvel)

        ind = np.int(slave.nvec/2)            #First guess
        spos = np.array(slave.pos[ind])
        svel = np.array(slave.vel[ind])
        svel = np.linalg.norm(svel)

        dx = spos - mpos;
        z_offset = slave.prf*np.dot(dx, vvec)/mvel

        ind = np.int(ind - z_offset)    #Refined estimate
        spos = slave.pos[ind]
        svel = slave.vel[ind]
        svel = np.linalg.norm(svel)

        dx = spos-mpos
        hb = np.dot(dx, crp)
        vb = np.dot(dx, rvec)

        csb = -1.0*hb*self.clook + vb*self.slook

#        print('Estimated Baseline: %4.2f'%csb)
        return csb


def parse():

    #    class RangeObj(object):
#       '''Class to deal with input ranges.'''
#       def __init__(self, start, end):
#           self.start = start
#           self.end = end
#       def __eq__(self, other):
#           return self.start <= other <= self.end


    def Range(nmin, nmax):
        class RangeObj(argparse.Action):
            def __call__(self, parser, args, values, option_string=None):
                if not nmin <= values <= nmax:
                    msg = 'Argument "{f}" requires value between {nmin} and {nmax}'.format(f=self.dest, nmin=nmin, nmax=nmax)
                    raise argparse.ArgumentTypeError(msg)
                setattr(args, self.dest, values)

        return RangeObj

    #####Actual parser set up
    parser = argparse.ArgumentParser(description='Computes the baseline plot for given set of SAR images.')
    parser.add_argument('fnames', nargs='+', default=None, help = 'XML files corresponding to the SAR scenes.')
    parser.add_argument('-Bcrit', dest='Bcrit', default=1200.0, help='Critical Geometric Baseline in meters [0., 10000.]', type=float, action=Range(0., 10000.))
    parser.add_argument('-Tau', dest='Tau', default=1080.0, help='Temporal Decorrelation Time Constant in days [0., 3650.]', type=float, action=Range(0., 3650.))
    parser.add_argument('-dop', dest='dop', default=0.5, help='Critical Doppler difference in fraction of PRF', type=float, action=Range(0., 1.))
    parser.add_argument('-coh', dest='cThresh', default=0.3, help='Coherence Threshold to estimate viable interferograms. [0., 1.0]', type=float, action=Range(0., 1.))
    parser.add_argument('-dir', dest='dirname', default='insar_XML', help='Directory in which the individual insar XML files are created.', type=str, action='store')
    parser.add_argument('-base', dest='base', default='base.xml', help='Base XML for the insar.xml files.', type=str)
    inps = parser.parse_args()

    return inps

if __name__ == '__main__':
    inps = parse()
    nSar = len(inps.fnames)
    print(inps.fnames)
    print('Number of SAR Scenes = %d'%nSar)

    Orbits = []
    print('Reading in all the raw files and metadata.')
    for k in xrange(nSar):
        sar = makeRawApp()
        sar.run(inps.fnames[k])
        Orbits.append(orbit_info(sar, inps.fnames[k]))

    ##########We now have all the pegpoints to start processing.
    Dopplers = np.zeros(nSar)
    Bperp    = np.zeros(nSar)
    Days     = np.zeros(nSar)

    #######Setting the first scene as temporary reference.
    master = Orbits[0]


    Dopplers[0] = master.fd
    Days[0] = master.dt.toordinal()
    for k in xrange(1,nSar):
        slave = Orbits[k]
        Bperp[k] = master.getBaseline(slave)
        Dopplers[k] = slave.fd
        Days[k]  = slave.dt.toordinal()


    print("************************************")
    print("Index    Date       Bperp  Doppler")
    print("************************************")

    for k in xrange(nSar):
        print('{0:>3}    {1:>10} {2:4.2f}  {3:4.2f}'.format(k+1, Orbits[k].dt.strftime('%Y-%m-%d'), Bperp[k],Dopplers[k]))


    print("************************************")

    geomRho = (1-np.clip(np.abs(Bperp[:,None]-Bperp[None,:])/inps.Bcrit, 0., 1.))
    tempRho = np.exp(-1.0*np.abs(Days[:,None]-Days[None,:])/inps.Tau)
    dopRho  = (np.abs(Dopplers[:,None] - Dopplers[None,:])/ master.prf) < inps.dop

    Rho = geomRho * tempRho * dopRho
    for kk in xrange(nSar):
        Rho[kk,kk] = 0.


    avgRho = np.mean(Rho, axis=1)*nSar/(nSar-1)
    numViable = np.sum((Rho> inps.cThresh), axis=1)

    ####Currently sorting on average coherence.

    masterChoice = np.argsort(avgRho)
    masterOrbit = Orbits[masterChoice[0]]
    masterBperp = Bperp[masterChoice[0]]


    print('*************************************')
    print('Ranking for Master Scene Selection: ')
    print('**************************************')
    print('Rank  Index      Date    nViable   Avg. Coh.' )
    for kk in xrange(nSar):
        ind = masterChoice[kk]
        print('{0:>3}   {1:>3}   {2:>10}  {3:>4}        {4:>2.3f}'.format(kk+1, ind+1, Orbits[ind].dt.strftime('%Y-%m-%d'), numViable[ind], avgRho[ind]))

    print('***************************************')

    print('***************************************')
    print('List of Viable interferograms:')
    print('***************************************')

#    if not os.path.isdir(inps.dirname):
#       try:
#           os.mkdir(inps.dirname)
#       except:
#           raise OSError("%s Directory cannot be created"%(inps.dirname))



    [ii,jj] = np.where(Rho > inps.cThresh)

    print('Master     Slave      Bperp      Deltat')
    for mind, sind in itertools.izip(ii,jj):
        master = Orbits[mind]
        slave = Orbits[sind]
        if master.dt > slave.dt:
            print('{0:>10} {1:>10}  {2:>4.2f}   {3:>4.2f}'.format(master.dt.strftime('%Y-%m-%d'), slave.dt.strftime('%Y-%m-%d'), Bperp[mind]-Bperp[sind], Days[mind] - Days[sind]))
            xmlname = '%s/insar_%s_%s.xml'%(inps.dirname, master.dt.strftime('%Y%m%d'), slave.dt.strftime('%Y%m%d'))

#           sarxml.sartoinsarXML(master.filename, slave.filename, base=inps.base, out=xmlname)


    print('***************************************')

    #######Currently picks master peg point.
    print('***************************************')
    commonPeg = masterOrbit.peg
    print('Common peg point:                      ')
    print(commonPeg)
    print('Bperp Range:  [%f , %f] '%(Bperp.min()-masterBperp, Bperp.max()-masterBperp))

    ######Choose median doppler
    commonDop = np.median(Dopplers)
    maxDop   = np.max(Dopplers)
    minDop = np.min(Dopplers)
    varDop = np.max(np.abs(Dopplers-commonDop))/masterOrbit.prf

    print('Common Doppler: ', commonDop)
    print('Doppler Range:  [%f, %f]'%(minDop, maxDop))
    print('MAx Doppler Variation = %f %%'%(varDop*100))
    print('******************************************')
