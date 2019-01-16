#!/usr/bin/env python
import os
import tempfile
import isce
import numpy as np 
import datetime
from isceobj.Orbit.Orbit import StateVector, Orbit
from collections import OrderedDict

try:
    import SpiceyPy
except ImportError:
    raise Exception('SpiceyPy Python bindings need to be installed to be able to use this library.')

class SpiceDatabase(object):
    '''
    Class for dealing with SPICE kernel files.
    '''

    dbdir  = os.path.join(os.path.dirname(__file__), 'db')
    dblist = os.path.join(dbdir, 'kernels.list')

    def __init__(self):
        '''
        Load the databasename.
        '''
        rdict = OrderedDict()
        infile = open(self.dblist, 'r')
        line = infile.readline()
        while line:
            llist = line.split('=')
            if len(llist)==2 :
                rdict[llist[0].strip()] = os.path.join(self.dbdir, llist[1].strip())
            line = infile.readline()
        infile.close()
        self.data = rdict

    def getList(self):
        '''Get list of kernel files.'''

        ll = []
        for key, val in self.data.items():
            ll.append(val)

        return ll

    def getKernel(self, key):
        return self.data[key]

    def __getitem__(self, key):
        return self.data[key]


class ISCEOrbit(object):
    '''
    Class for converting ISCE orbits to CSPICE orbits.
    '''

    def __init__(self, orbit):
        self.orbit = orbit
        self.db = SpiceDatabase()

    def exportToSPK(self, spkfile,frame='ITRF93'):
        '''
        Export ISCE orbit to SPK file.
        '''

        if frame not in ('ITRF93', 'J2000', 'ECI_TOD', 'ECLIPJ2000'):
            raise Exception('CSPICE currently only supports ITRF93, J2000, ECLIPJ2000, ECI_TOD.')

        tmpDir = tempfile.mkdtemp(dir='.')

        hdrfile = os.path.join(tmpDir, 'hdrfile')
        setupfile = os.path.join(tmpDir, 'setupfile')
        self.exportOrbitToHeader(hdrfile)
        self.createSetupFile(setupfile, frame=frame)
        self.runMkspk(hdrfile, setupfile, spkfile)

        for root, dirs, files in os.walk(tmpDir):
            for filename in files:
                try:
                    os.unlink(os.path.join(tmpDir, filename))
                except:
                    os.system("rm "+os.path.join(tmpDir, filename))

        os.rmdir(tmpDir)


    def exportOrbitToHeader(self, hdrfile):
        '''
        Exports a given Orbit to SPICE compatible HDR format.
        '''

        fid = open(hdrfile, 'w')
        for sv in self.orbit:
            tim = sv.getTime()
            pos = sv.getPosition()
            vel = sv.getVelocity()
            pos =[str(x/1000.) for x in pos]
            vel = [str(x/1000.) for x in vel]

            out = [str(tim)] + pos + vel
            fid.write(','.join(out) + '\n')

        fid.close()
   
    def createSetupFile(self, setupfile, frame=None):
        '''
        Creates a setup file to use with mkspk.
        '''
        
        fmtString =  """\\begindata
INPUT_DATA_TYPE     = 'STATES'
OUTPUT_SPK_TYPE     = 13
OBJECT_ID           = -123710
OBJECT_NAME         = 'RADARSAT'
CENTER_ID           = 399
CENTER_NAME         = 'EARTH'
REF_FRAME_NAME      = '{0}'
PRODUCER_ID         = 'ISCE py3'
DATA_ORDER          = 'EPOCH X Y Z VX VY VZ'
INPUT_DATA_UNITS    = ( 'ANGLES=DEGREES' 'DISTANCES=km')
DATA_DELIMITER      = ','
LINES_PER_RECORD    = 1
LEAPSECONDS_FILE    = '{1}'
POLYNOM_DEGREE      = 3
SEGMENT_ID          = 'SPK_STATES_13'
TIME_WRAPPER        = '# UTC'
PCK_FILE            = ('{2}')

"""

        frameString = """FRAME_DEF_FILE      = ('{0}')

"""

        txtString="\\begintext"

        tfirst = self.orbit._stateVectors[0].getTime()

        leap = self.db['LEAPSECONDS']

        if tfirst.date() < datetime.date(2000,1,1):
            pck = self.db['EARTHHIGHRES']
        else:
            pck = self.db['EARTHHIGHRESLATEST']
        tod = self.db['EARTHECI_TOD']

        #####Link them to temp dir
        tmpdir = os.path.dirname(setupfile)
        leaplnk = os.path.join(tmpdir, os.path.basename(leap))
        try:
            os.link(leap, leaplnk)
        except:
            os.system("ln -s "+leap+" "+leaplnk)

        pcklnk = os.path.join(tmpdir, os.path.basename(pck))
        try:
            os.link(pck, pcklnk)
        except:
            os.system("ln -s "+pck+" "+pcklnk)


        if frame == 'ECI_TOD':
            todlnk = os.path.join(tmpdir, os.path.basename(tod))
            try:
                os.link(tod, todlnk)
            except:
                os.system("ln -s "+tod+" "+todlnk)

        outstr = fmtString.format(frame, leaplnk, pcklnk)

        if frame == 'ECI_TOD':
            outstr = outstr + frameString.format(todlnk)

        outstr = outstr + txtString

        fid = open(setupfile, 'w')
        fid.write(outstr)
        fid.close()


    def runMkspk(self, hdrfile, setupfile, spkfile):
        if os.path.exists(spkfile):
            print('Removing old version of spk file')
            os.remove(spkfile)

        cmd = ['mkspk', '-input '+hdrfile,
                '-setup '+ setupfile, '-output ' + spkfile]
        os.system(' '.join(cmd))

        pass


class SpiceOrbit(object):
    '''
    Orbit for dealing with Spice bsp files.
    '''

    def __init__(self, spkfile):
        '''
        Constructor.
        '''
        self.spkfile = spkfile
        self.db = SpiceDatabase()

    def initSpice(self):
        ll = self.db.getList()
        for val in ll:
            SpiceyPy.furnsh(val)

        SpiceyPy.furnsh(self.spkfile)

    def interpolateOrbit(self, time, frame='ITRF93'):

        if frame not in ('ITRF93', 'J2000', 'ECI_TOD', 'ECLIPJ2000'):
            raise Exception('Currently only ITRF93/J2000 frames are supported.')
        et = SpiceyPy.str2et(str(time))
        res,lt = SpiceyPy.spkezr('-123710', et,
                frame, 'None', 'EARTH')
        sv = StateVector()
        sv.setTime(time)
        sv.setPosition([x*1000.0 for x in res[0:3]])
        sv.setVelocity([x*1000.0 for x in res[3:6]])
        return sv

def loadHdrAsOrbit(fname, date=None):
    '''Read a hdr file and convert to ISCE orbit'''
    from isceobj.Orbit.Orbit import Orbit, StateVector

    if date is None:
        date = datetime.datetime.now().date()

    t0 = datetime.datetime(year=date.year,
                        month = date.month,
                        day = date.day)
    orb = Orbit()
    inData = np.loadtxt(fname)

    for line in inData:
        time = t0 + datetime.timedelta(seconds = line[0])
        sv = StateVector()
        sv.setTime(time)
        sv.setPosition(line[1:4].tolist())
        sv.setVelocity(line[4:7].tolist())
        orb.addStateVector(sv)
        print(sv)

    return orb


def dumpOrbitToHdr(orbit, filename, date=None):
    '''
    Dump orbit to ROI_PAC style hdr file.
    '''

    if date is None:
        date = orbit._stateVectors[0].getTime().date()

    t0 = datetime.datetime(year = date.year,
                           month = date.month,
                           day = date.day)
    
    nVec = len(orbit._stateVectors)
    arr = np.zeros((nVec, 7))

    for ind,sv in enumerate(orbit._stateVectors):
        arr[ind][0] = (sv.getTime() - t0).total_seconds()
        arr[ind][1:4] = sv.getPosition()
        arr[ind][4:] = sv.getVelocity()

    
    np.savetxt(filename, arr, fmt='%10.6f')


class ECI2ECEF(object):
    '''
    Class for converting Inertial orbits to ECEF orbits using JPL's SPICE library.
    '''

    def __init__(self, orbit, eci='J2000', ecef='ITRF93'):
        '''
        Currently J2000 and ITRF3 frames are supported by the SPICE library.
        '''
        
        self.eci  = eci
        self.ecef = ecef
        self.orbit = orbit

    def convert(self):
        '''Convert ECI orbit to ECEF orbit.'''

        date = self.orbit._stateVectors[0].getTime().date()
        bspName = 'inertial_orbit_' + date.strftime('%Y%m%d') + '.bsp'

        ####Convert ISCE orbit to SPICE orbit file
        sorb = ISCEOrbit(self.orbit)
        sorb.exportToSPK(bspName, frame=self.eci)

        ####Convert coordinates with corrections
        spk = SpiceOrbit(bspName)
        spk.initSpice()

        wgsOrbit = Orbit()
        wgsOrbit.setOrbitSource( self.orbit.getOrbitSource())
        wgsOrbit.setOrbitQuality( self.orbit.getOrbitQuality()) 
        for sv in self.orbit:
            tim = sv.getTime()
            spksv = spk.interpolateOrbit(tim, frame=self.ecef)
            wgsOrbit.addStateVector(spksv)

        return wgsOrbit
