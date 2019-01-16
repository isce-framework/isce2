#!/usr/bin/env python3

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
import datetime
import isceobj.Sensor.CEOS as CEOS
import logging
from isceobj.Orbit.Orbit import StateVector,Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from iscesys.Component.Component import Component
from isceobj.Sensor import xmlPrefix
from isceobj.Util import Polynomial
from iscesys.DateTimeUtil import secondsSinceMidnight
import numpy as np
import struct

INPUT_DIRECTORY_LIST = Component.Parameter(
        'inputDirList',
        public_name='input directory',
        type = str,
        container = list,
        mandatory = True,
        doc = 'List of input directories to parse')

INPUT_SWATH_LIST = Component.Parameter(
        'swaths',
        public_name='swaths',
        type=int,
        container=list,
        mandatory=False,
        default=None,
        doc = 'List of swaths to use')

POLARIZATION = Component.Parameter(
        'polarization',
        public_name='polarization',
        type=str,
        default='hh',
        mandatory=False,
        doc='Polarization to search for')

VIRTUAL_FILES = Component.Parameter(
        'virtualFiles',
        public_name='use virtual files',
        type=bool,
        default=True,
        mandatory=False,
        doc='Use virtual files instead of using disk space')

OUTPUTDIR = Component.Parameter(
        'output',
        public_name='output directory',
        type = str,
        default=None,
        mandatory = True,
        doc = 'Output directory for unpacking the data')


MAX_SWATHS = Component.Parameter(
        'maxSwaths',
        public_name='maximum number of swaths',
        type=int,
        default=5,
        mandatory=True,
        doc = 'Maximum number of swaths to scan for')

####ALOS2 directory browser
class ALOS2Scanner(Component):

    family = 'alos2scanner'

    parameter_list = (INPUT_DIRECTORY_LIST,
                      INPUT_SWATH_LIST,
                      POLARIZATION,
                      VIRTUAL_FILES,
                      OUTPUTDIR,
                      MAX_SWATHS)

    modes = ['WBS', 'WBD', 'WWS', 'WWD', 'VBS', 'VBD']


    def __init__(self, name=''):
        super(ALOS2Scanner, self).__init__(family=self.__class__.family, name=name)


    def scan(self):

        if isinstance(self.inputDirList, str):
            self.inputDirList = [self.inputDirList]

        frames = []
        for indir in self.inputDirList:
            frames.append( self.scanDir(indir))

        if len(frames) == 0:
            raise Exception('No products found in the input directories')

        ###Estimate common swaths
        return frames  


    def extractImage(self):
        '''
        Actual extraction of SLCs.
        '''

        totalSwaths = []

        frames = self.scan()

        ###Currently assuming one frame
        ###Modify here for multiple frames
        for swathid, img in frames[0].items():

            sensor = ALOS2()
            sensor.configure()
            sensor._leaderFile = img['leaderfile']
            sensor._imageFile = img['imgfile']

            outdir = os.path.join(self.output, img['frame'], 's{0}'.format(swathid))
            sensor.output = os.path.join(outdir, 'swath.slc')

            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            sensor.extractImage(virtual=self.virtualFiles)
            sensor.extractDoppler()
            sensor.refineBurstTiming()

            totalSwaths.append(sensor.frame)

        return totalSwaths

    def scanDir(self, indir):
        '''
        Scan directory for IMG files.
        '''

        import glob
        import os

        imgFiles = glob.glob(os.path.join(indir, 'IMG-{0}-ALOS2*-*-*'.format(self.polarization.upper())))
        ###No IMG files found 
        if len(imgFiles) == 0:
            return None

        ###Sort the filenames
        imgFiles = sorted(imgFiles)

        #######
        wbFiles = []
        for infile in imgFiles:
            basefile = os.path.basename(infile)
            
            ##Check for each mode
            for mode in self.modes:
                if mode in basefile:
                    wbFiles.append(infile)
                    break
        
        if len(wbFiles) == 0:
            return None

        ###Check if user has requested specific files
        frames = []
        datatakes = []
        imgmodes = []
        for infile in wbFiles:
            basefile = os.path.basename(infile)
            frames.append( basefile.split('-')[2][-4:])
            datatakes.append( basefile.split('-')[2][5:10])
            imgmodes.append( basefile.split('-')[-2][0:3])

        if any([x!=frames[0] for x in frames]):
            print('Multiple frames found in same dir')
            print(set(frames))
            raise Exception('Multiple ALOS2 frames in same dir')

        if any([x!=datatakes[0] for x in datatakes]):
            print('Multiple datatakes found in same dir')
            print(set(datatakes))
            raise Exception('Multiple ALOS2 datatakes found in same dir')

        if any([x!=imgmodes[0] for x in imgmodes]):
            print('Multiple imaging modes found in same dir')
            print(set(imgmodes))
            raise Exception('Multiple ALOS2 imaging modes found in same dir')

        swaths = {}
        for infile in wbFiles:
            params = {}
            params['datatake'] = datatakes[0]
            params['frame'] = frames[0]
            params['imgfile'] = infile

            swathid = int(os.path.basename(infile)[-1])

            ##If user has requested specific swaths
            if self.swaths:
                if swathid in self.swaths:
                    swaths[swathid] = params
            else:
                swaths[swathid] = params


        ###Ensure that a LED file exists that matches the data
        ldrfiles = glob.glob(os.path.join(indir, 'LED-ALOS2{0}{1}-*'.format(datatakes[0], frames[0])))
        if len(ldrfiles) == 0:
            raise Exception('No leader file found in ALOS2 directory')

        if len(ldrfiles) > 1:
            raise Exception('More than one leader file found in ALOS2 directory')

        leaderFile = ldrfiles[0]
        
        for key, val in swaths.items():
            swaths[key]['leaderfile'] = leaderFile

        return swaths

        





#####Actual ALOS reader
#Sometimes the wavelength in the meta data is not correct.
#If the user sets this parameter, then the value in the
#meta data file is ignored.
WAVELENGTH = Component.Parameter(
    'wavelength',
    public_name='radar wavelength',
    default=None,
    type=float,
    mandatory=False,
    doc='Radar wavelength in meters.'
)

LEADERFILE = Component.Parameter(
    '_leaderFile',
    public_name='leaderfile',
    default=None,
    type=str,
    mandatory=True,
    doc='Name of the leaderfile.'
)

IMAGEFILE = Component.Parameter(
    '_imageFile',
    public_name='imagefile',
    default=None,
    type=str,
    mandatory=True,
    doc='Name of the imagefile.'
)

OUTPUT = Component.Parameter('output',
        public_name='OUTPUT',
        default=None,
        type=str,
        doc = 'Directory where bursts get unpacked') 


###List of facilities
FRAME = Component.Facility('frame',
                public_name = 'frame',
                module = 'isceobj.Sensor.ScanSAR',
                factory = 'createFullApertureSwathSLCProduct',
                args = (),
                mandatory=True,
                doc = 'Full aperture swath slc product populated by the reader')


class ALOS2(Component):
    """
        Code to read CEOSFormat leader files for ALOS2 SLC data.
    """

    family = 'alos2'

    parameter_list = (WAVELENGTH,
                      LEADERFILE,
                      IMAGEFILE,
                      OUTPUT)

    facility_list = (FRAME,)
    fsampConst = { 104: 1.047915957140240E+08,
                   52: 5.239579785701190E+07,
                   34: 3.493053190467460E+07,
                   17: 1.746526595233730E+07 }

    #Orbital Elements (Quality) Designator
    #ALOS-2/PALSAR-2 Level 1.1/1.5/2.1/3.1 CEOS SAR Product Format Description
    #PALSAR-2_xx_Format_CEOS_E_r.pdf
    orbitElementsDesignator = {'0':'preliminary',
                               '1':'decision',
                               '2':'high precision'}

    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name)
        self.leaderFile = None
        self.imageFile = None

        #####Soecific doppler functions for ALOS2
        self.doppler_coeff = None
        self.azfmrate_coeff = None
        self.lineDirection = None
        self.pixelDirection = None

        self.constants = {'polarization': 'HH',
                          'antennaLength': 10}


    def getFrame(self):
        return self.frame

    def parse(self):
        self.leaderFile = LeaderFile(self, file=self._leaderFile)
        self.leaderFile.parse()

        self.imageFile = ImageFile(self, file=self._imageFile)
        self.imageFile.parse()

        self.populateMetadata()

    def populateMetadata(self):
        """
            Create the appropriate metadata objects from our CEOSFormat metadata
        """
        frame = self._decodeSceneReferenceNumber(self.leaderFile.sceneHeaderRecord.metadata['Scene reference number'])

        fsamplookup = int(self.leaderFile.sceneHeaderRecord.metadata['Range sampling rate in MHz'])

        rangePixelSize = Const.c/(2*self.fsampConst[fsamplookup])

        ins = self.frame.getInstrument()
        platform = ins.getPlatform()
        platform.setMission(self.leaderFile.sceneHeaderRecord.metadata['Sensor platform mission identifier'])
        platform.setAntennaLength(self.constants['antennaLength'])
        platform.setPointingDirection(1)
        platform.setPlanet(Planet(pname='Earth'))

        if self.wavelength:
            ins.setRadarWavelength(float(self.wavelength))
#            print('ins.radarWavelength = ', ins.getRadarWavelength(),
#                  type(ins.getRadarWavelength()))
        else:
            ins.setRadarWavelength(self.leaderFile.sceneHeaderRecord.metadata['Radar wavelength'])

        ins.setIncidenceAngle(self.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre'])
        self.frame.getInstrument().setPulseRepetitionFrequency(self.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency in mHz']*1.0e-3)
        ins.setRangePixelSize(rangePixelSize)
        ins.setRangeSamplingRate(self.fsampConst[fsamplookup])
        ins.setPulseLength(self.leaderFile.sceneHeaderRecord.metadata['Range pulse length in microsec']*1.0e-6)
        chirpSlope = self.leaderFile.sceneHeaderRecord.metadata['Nominal range pulse (chirp) amplitude coefficient linear term']
        chirpPulseBandwidth = abs(chirpSlope * self.leaderFile.sceneHeaderRecord.metadata['Range pulse length in microsec']*1.0e-6)
        ins.setChirpSlope(chirpSlope)
        ins.setInPhaseValue(7.5)
        ins.setQuadratureValue(7.5)

        self.lineDirection = self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along line direction'].strip()
        self.pixelDirection =  self.leaderFile.sceneHeaderRecord.metadata['Time direction indicator along pixel direction'].strip()

        ######ALOS2 includes this information in clock angle
        clockAngle = self.leaderFile.sceneHeaderRecord.metadata['Sensor clock angle']
        if clockAngle == 90.0:
            platform.setPointingDirection(-1)
        elif clockAngle == -90.0:
            platform.setPointingDirection(1)
        else:
            raise Exception('Unknown look side. Clock Angle = {0}'.format(clockAngle))

#        print(self.leaderFile.sceneHeaderRecord.metadata["Sensor ID and mode of operation for this channel"])
        self.frame.setFrameNumber(frame)
        self.frame.setOrbitNumber(self.leaderFile.sceneHeaderRecord.metadata['Orbit number'])
        self.frame.setProcessingFacility(self.leaderFile.sceneHeaderRecord.metadata['Processing facility identifier'])
        self.frame.setProcessingSystem(self.leaderFile.sceneHeaderRecord.metadata['Processing system identifier'])
        self.frame.setProcessingSoftwareVersion(self.leaderFile.sceneHeaderRecord.metadata['Processing version identifier'])
        self.frame.setPolarization(self.constants['polarization'])
        self.frame.setNumberOfLines(self.imageFile.imageFDR.metadata['Number of lines per data set'])
        self.frame.setNumberOfSamples(self.imageFile.imageFDR.metadata['Number of pixels per line per SAR channel'])

        ######
        orb = self.frame.getOrbit()

        orb.setOrbitSource('Header')
        orb.setOrbitQuality(
            self.orbitElementsDesignator[
                self.leaderFile.platformPositionRecord.metadata['Orbital elements designator']
            ]
        )
        t0 = datetime.datetime(year=self.leaderFile.platformPositionRecord.metadata['Year of data point'],
                               month=self.leaderFile.platformPositionRecord.metadata['Month of data point'],
                               day=self.leaderFile.platformPositionRecord.metadata['Day of data point'])
        t0 = t0 + datetime.timedelta(seconds=self.leaderFile.platformPositionRecord.metadata['Seconds of day'])

        #####Read in orbit in inertial coordinates
        deltaT = self.leaderFile.platformPositionRecord.metadata['Time interval between data points']
        numPts = self.leaderFile.platformPositionRecord.metadata['Number of data points']


        orb = self.frame.getOrbit()
        for i in range(numPts):
            vec = StateVector()
            t = t0 + datetime.timedelta(seconds=i*deltaT)
            vec.setTime(t)

            dataPoints = self.leaderFile.platformPositionRecord.metadata['Positional Data Points'][i]
            pos = [dataPoints['Position vector X'], dataPoints['Position vector Y'], dataPoints['Position vector Z']]
            vel = [dataPoints['Velocity vector X'], dataPoints['Velocity vector Y'], dataPoints['Velocity vector Z']]
            vec.setPosition(pos)
            vec.setVelocity(vel)
            orb.addStateVector(vec)


        ###This is usually available with ALOS SLC data.
        ###Unfortunately set to all zeros for ScanSAR data
        #self.doppler_coeff = [self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid constant term'],
        #self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid linear term'],
        #self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency centroid quadratic term']]


        self.azfmrate_coeff =  [self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate constant term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate linear term'],
        self.leaderFile.sceneHeaderRecord.metadata['Cross track Doppler frequency rate quadratic term']]


        ###Reading in approximate values instead
        ###Note that these are coeffs vs slant range in km
        self.doppler_coeff = [self.leaderFile.sceneHeaderRecord.metadata['Doppler center frequency constant term'],
                             self.leaderFile.sceneHeaderRecord.metadata['Doppler center frequency linear term']]

#        print('Terrain height: ', self.leaderFile.sceneHeaderRecord.metadata['Average terrain ellipsoid height'])


    def extractImage(self, virtual=False):
        import isceobj
        if (self.imageFile is None) or (self.leaderFile is None):
            self.parse()

        
        ###Generating XML file first as renderHdr also creates a VRT
        ###We want the virtual CEOS VRT to overwrite the general style VRT
        rawImage = isceobj.createSlcImage()
        rawImage.setByteOrder('l')
        rawImage.setFilename(self.output)
        rawImage.setAccessMode('read')
        rawImage.setWidth(self.imageFile.width)
        rawImage.setLength(self.imageFile.length)
        rawImage.setXmin(0)
        rawImage.setXmax(self.imageFile.width)
        rawImage.renderHdr()
        self.frame.setImage(rawImage)

        self.imageFile.extractImage(output=self.output, virtual=virtual)

        self.frame.setSensingStart(self.imageFile.sensingStart)
        self.frame.setSensingStop(self.imageFile.sensingStop)
        sensingMid = self.imageFile.sensingStart + datetime.timedelta(seconds = 0.5* (self.imageFile.sensingStop - self.imageFile.sensingStart).total_seconds())
        self.frame.setSensingMid(sensingMid)

        self.frame.setStartingRange(self.imageFile.nearRange)

        self.frame.getInstrument().setPulseRepetitionFrequency(self.imageFile.prf)

        pixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = self.imageFile.nearRange + (pixelSize-1) * self.imageFile.width
        self.frame.setFarRange(farRange)

        return


    def extractDoppler(self):
        '''
        Evaluate the doppler and fmrate polynomials.
        '''
        import copy

        ##We typically use this ALOS2 SLCs
        ####CEOS already provides function vs pixel
        #self.frame._dopplerVsPixel = self.doppler_coeff
        

        ##Instead for ScanSAR data, we have to do the mapping from approx coeffs
        frame = self.frame
        width = frame.getNumberOfSamples()
        rng = frame.startingRange + np.arange(0,width,100) * 0.5 * Const.c/frame.rangeSamplingRate
        doppler = self.doppler_coeff[0] + self.doppler_coeff[1] * rng/1000.
        dfit = np.polyfit( np.arange(0, width, 100), doppler, 1) 
        self.frame._dopplerVsPixel=[dfit[1], dfit[0], 0., 0.]

        ##We have to compute FM rate here.
        ##Cunren's observation that this is all set to zero in CEOS file.
        ##Simplification from Cunren's fmrate.py script
        ##Should be the same as the one in focus.py
        planet = self.frame.instrument.platform.planet
        elp = copy.copy(planet.ellipsoid)
        svmid = self.frame.orbit.interpolateOrbit(self.frame.sensingMid, method='hermite')
        xyz = svmid.getPosition()
        vxyz = svmid.getVelocity()
        llh = elp.xyz_to_llh(xyz)
        hdg = self.frame.orbit.getENUHeading(self.frame.sensingMid)

        elp.setSCH(llh[0], llh[1], hdg)
        sch, schvel = elp.xyzdot_to_schdot(xyz, vxyz)

        ##Computeation of acceleration
        dist= np.linalg.norm(xyz)
        r_spinvec = np.array([0., 0., planet.spin])
        r_tempv = np.cross(r_spinvec, xyz)
        inert_acc = np.array([-planet.GM*x/(dist**3) for x in xyz])
        r_tempa = np.cross(r_spinvec, vxyz)
        r_tempvec = np.cross(r_spinvec, r_tempv)
        axyz = inert_acc - 2 * r_tempa - r_tempvec
        
        schbasis = elp.schbasis(sch)
        schacc = np.dot(schbasis.xyz_to_sch, axyz).tolist()[0]


        ##Jumping back straight into Cunren's script here
        centerVel = schvel
        centerAcc = schacc
        avghgt = llh[2]
        radiusOfCurvature = elp.pegRadCur
        frame = self.frame

        fmrate = []
        width = self.frame.getNumberOfSamples()
        lookSide = self.frame.instrument.platform.pointingDirection
        centerVelNorm = np.linalg.norm(centerVel)

        ##Retaining Cunren's code for computing at every pixel.
        ##Can be done every 10th pixel since we only fit a quadratic/ cubic.
        ##Also can be vectorized for speed.

        for ii in range(width):
            rg = frame.startingRange + ii * 0.5 * Const.c / frame.rangeSamplingRate
            dop = np.polyval(frame._dopplerVsPixel[::-1], ii)

            th = np.arccos(((avghgt+radiusOfCurvature)**2 + rg**2 -radiusOfCurvature**2)/(2.0 * (avghgt + radiusOfCurvature) * rg))
            thaz = np.arcsin(((frame.radarWavelegth*dop/(2.0*np.sin(th))) + (centerVel[2] / np.tan(th))) / np.sqrt(centerVel[0]**2 + centerVel[1]**2)) - lookSide * np.arctan(centerVel[1]/centerVel[0])

            lookVec = [ np.sin(th) * np.sin(thaz),
                        np.sin(th) * np.cos(thaz) * lookSide,
                        -np.cos(th)]

            vdotl = np.dot(lookVec, centerVel)
            adotl = np.dot(lookVec, centerAcc)
            fmratex = 2.0*(adotl + (vdotl**2 - centerVelNorm**2)/rg)/(frame.radarWavelegth)
            fmrate.append(fmratex)


        ##Fitting order 2 polynomial to FM rate
        p = np.polyfit(np.arange(width), fmrate,2)
        frame._fmrateVsPixel = list(p[::-1])


    def _decodeSceneReferenceNumber(self,referenceNumber):
        return referenceNumber

    
    def refineBurstTiming(self):
        '''
        This is combination of burst_time2.py and burst_time.py from Cunren.
        '''

        slc = self.frame.image.filename

        ##First pass of burst_time.py
        delta_line = 15000
        bursts1 = self.burst_time( slc,
                        firstLine=delta_line, firstPixel=1000)

        
        ##Number of burst cycles
        num_nc = np.around((self.frame.getNumberOfLines() - delta_line*2)/ self.frame.ncraw)

        ###Second pass
        start_line2 = np.around( delta_line + num_nc * self.frame.ncraw)
        bursts2 = self.burst_time( slc,
                        firstLine=start_line2, firstPixel=1000)

        ###Check if there were differences
        LineDiffIndex = 0
        LineDiffMin = np.fabs( bursts1['estimatedStartLine'] + self.frame.ncraw * LineDiffIndex - bursts2['estimatedStartLine'])

        for ii in range(100000):
            LineDiffMinx = np.fabs(bursts1['estimatedStartLine'] + self.frame.ncraw * ii - bursts2['estimatedStartLine'])
            if LineDiffMinx <= LineDiffMin:
                LineDiffMin = LineDiffMinx
                LineDiffIndex = ii

        ###Update correct burst cycle value
        print('Burst cycle length before correction: ', self.frame.ncraw)
        self.frame.ncraw = self.frame.ncraw - (bursts1['estimatedStartLine'] + self.frame.ncraw * LineDiffIndex - bursts2['estimatedStartLine'])/LineDiffIndex
        print('Burst cycle length after correction: ', self.frame.ncraw)


        ###Final run with updated burst cycle length
        start_line1 = np.around(self.frame.getNumberOfLines() / 2.0)
        bursts = self.burst_time( slc, 
                        firstLine=start_line1, firstPixel=1000)

        self.frame.burstStartLines = bursts['startLines']
        for ii, val in enumerate(self.frame.burstStartLines):
            print('Burst: {0}, Line: {1}'.format(ii, val))

    def burst_time(self, slcfile, 
            firstLine=500, firstPixel=500,
            nRange=400):
        '''
        Generates a linear FM signal and returns correlation with signal.
        '''

        def create_lfm(ns, it, offset, k):
            '''
            Create linear FM signal.
            ns: Number of samples
            it: Time interval of samples
            offset: offset
            k: linear FM rate
            '''
            ht = (ns-1)/2.0
            t = np.arange(-ht, ht+1.0, 1)
            t = (t + offset) * it
            lfm = np.exp(1j * np.pi * k * t**2)
            return lfm

        from osgeo import gdal

        frame = self.frame
        width = frame.getNumberOfSamples()
        length = frame.getNumberOfLines()
        prf = frame.PRF
        nb = frame.nbraw
        nc = frame.ncraw
        fmrateCoeff = frame._fmrateVsPixel
        sensing_start = frame.getSensingStart()
        
        ###Using convention that Fmrate is positive
        ka = -np.polyval(fmrateCoeff[::-1], np.arange(width))

        ###Area to be used for estimation
        saz = firstLine             #Startline to be included
        naz = int(np.round(nc))     #Number of lines to be used
        eaz = saz + naz-1           #Ending line to be used
        caz = int(np.round((saz+eaz)/2.0)) #Central line of lines used
        caz_deramp = (saz+eaz)/2.0   #Center of deramp signal

        srg = firstPixel            #Start column to be used
        nrg = nRange                #Number columns to be used
        erg = srg + nrg - 1         #Ending column to be used
        crg = int(np.round((srg+erg)/2.0)) #Central column


        if not (saz >=0 and saz <= length-1):
            raise Exception('Invalid starting line \n')

        if not (eaz >=0 and eaz <= length-1):
            raise Exception('Invalid ending line \n')

        if not (srg >= 0 and erg <= width-1):
            raise Exception('Invalid starting column \n')

        if not (erg >=0 and erg <= width-1):
            raise Exception('Invalid ending column \n')

        ###Calculate full aperture length
        nFullAperture = int(np.round(prf/ka[crg]/(1.0/prf)))
        nazfft = int(2**(int(np.ceil(np.log2(nFullAperture)))))

        ###Create the deramp function using fmrate
        deramp = np.zeros((naz,nrg), dtype=np.complex64)
        for ii in range(nrg):
            deramp[:,ii] = create_lfm(naz, 1.0/prf, 0, -ka[ii+srg])

        ###Read in chunk of data
        ds = gdal.Open(slcfile + '.vrt', gdal.GA_ReadOnly)
        data = ds.ReadAsArray(srg, saz, nrg, naz)
        ds = None

        ###deramp the data
        datadr = deramp * data

        #Compute spectrum
        spec = np.fft.fft(datadr, n=nazfft, axis=0)

        #Center the spectrum
        spec = np.fft.fftshift(spec, axes=0)

        ##Average the spectrum
        avgSpec = np.mean( np.abs(spec), axis=1)

        ###Number of bursts in freq domain
        nbs = int(np.round(nb*(1.0/prf)*ka[crg]/prf*nazfft))

        ###Number of samples of the burst cycle in frequency domain
        ncs = int(np.round(nc*(1.0/prf)*ka[crg]/prf*nazfft))

        ###Create a signal corresponding to 1 burst spectrum length
        rect = np.ones(nbs, dtype=np.float32)

        ##Correlated burst with average spectrum
        corr = np.correlate(avgSpec, rect, 'same')

        ###Find burst spectrum center
        ncs_rh = int(np.round((nazfft - ncs)/2.0))

        ##Offset between spectrum center and center
        offset_spec = np.argmax(corr[ncs_rh:ncs_rh+ncs]) + ncs_rh - (nazfft-1.0)/2.0
        
        ##Offset in azimuth lines
        offset_naz = offset_spec / nazfft * prf / ka[crg] / (1.0 / prf)

        ##Starting line of the burst (fractional line number)
        saz_burst = -offset_naz + caz_deramp - (nb-1.0)/2.0

        ####Find the start lines of all bursts
        burstStartLines = []
        burstStartTimes = []

        for ii in range(-100000, 100000):
            saz_burstx = saz_burst + nc * ii

            if (saz_burstx >= 0.0) and (saz_burstx <= length):
                st_burstx = sensing_start + datetime.timedelta(seconds=saz_burstx/prf)
                burstStartLines.append(saz_burstx)
                burstStartTimes.append(st_burstx)

        bursts = {}
        bursts['startLines'] = burstStartLines
        bursts['startTimes'] = burstStartTimes
        bursts['estimatedStartLine'] = saz_burst

        #for ii in range(len(bursts['startLines'])):
        #    print(ii, bursts['startLines'][ii], bursts['startTimes'][ii])

        return bursts



class LeaderFile(object):

    def __init__(self, parent, file=None):
        self.parent = parent
        self.file = file
        self.leaderFDR = None
        self.sceneHeaderRecord = None
        self.platformPositionRecord = None

    def parse(self):
        """
            Parse the leader file to create a header object
        """
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return
        # Leader record
        self.leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/leader_file.xml'),dataFile=fp)
        self.leaderFDR.parse()
        fp.seek(self.leaderFDR.getEndOfRecordPosition())

        # Scene Header
        self.sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/scene_record.xml'),dataFile=fp)
        self.sceneHeaderRecord.parse()
        fp.seek(self.sceneHeaderRecord.getEndOfRecordPosition())

        # Platform Position
        self.platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/platform_position_record.xml'),dataFile=fp)
        self.platformPositionRecord.parse()
        fp.seek(self.platformPositionRecord.getEndOfRecordPosition())

        #####Skip attitude information
        fp.seek(16384,1)

        #####Skip radiometric information
        fp.seek(9860,1)

        ####Skip the data quality information
        fp.seek(1620,1)


        ####Skip facility 1-4
        fp.seek(325000 + 511000 + 3072 + 728000, 1)


        ####Read facility 5
        self.facilityRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/facility_record.xml'), dataFile=fp)
        self.facilityRecord.parse()
        fp.close()

class VolumeDirectoryFile(object):

    def __init__(self,file=None):
        self.file = file
        self.metadata = {}

    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        volumeFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/volume_descriptor.xml'),dataFile=fp)
        volumeFDR.parse()
        fp.seek(volumeFDR.getEndOfRecordPosition())

        fp.close()


class ImageFile(object):

    def __init__(self, parent, file=None):
        self.parent = parent
        self.file = file
        self.imageFDR = None
        self.sensingStart = None
        self.sensingStop = None
        self.nearRange = None
        self.prf = None
        self.image_record = os.path.join(xmlPrefix,'alos2_slc/image_record.xml')
        self.logger = logging.getLogger('isce.sensor.alos2')

    def parse(self):
        try:
            fp = open(self.file,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        self.imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_file.xml'), dataFile=fp)
        self.imageFDR.parse()
        fp.seek(self.imageFDR.getEndOfRecordPosition())
        self._calculateRawDimensions(fp)

        fp.close()

    def writeRawData(self, fp, line):
        '''
        Convert complex integer to complex64 format.
        '''
        cJ = np.complex64(1j)
        data = line[0::2] + cJ * line[1::2]
        data.tofile(fp)


    def extractImage(self, output=None, virtual=False):
        """
        Extract I and Q channels from the image file
        """
        if virtual:
            output = output + '.vrt'
        else:
            try:
                output = open(output, 'wb')
            except IOError as strerr:
                raise Exceptin("IOError: {0}".format(strerr))


        
        if self.imageFDR is None:
            self.parse()


        ###Open the image file for reading
        try:
            fp = open(self.file, 'rb')
        except IOError as strerr:
            self.logger.error(" IOError: %s" % strerr)
            return


        fp.seek(self.imageFDR.getEndOfRecordPosition(),os.SEEK_SET)
        offsetAfterImageFDR = fp.tell()


        dataLen =  self.imageFDR.metadata['Number of pixels per line per SAR channel']
        self.width = dataLen

        ##Leaderfile PRF
        prf = self.parent.leaderFile.sceneHeaderRecord.metadata['Pulse Repetition Frequency in mHz']*1.0e-3

        #choose PRF according to operation mode. Cunren Liang, 2015
        operationMode = "{}".format(self.parent.leaderFile.sceneHeaderRecord.metadata['Sensor ID and mode of operation for this channel'])
        operationMode =operationMode[10:12]
        if operationMode not in ['08', '09']:
            # Operation mode
            # '00': Spotlight mode
            # '01': Ultra-fine
            # '02': High-sensitive
            # '03': Fine
            # '08': ScanSAR nominal mode
            # '09': ScanSAR wide mode
            # '18': Full (Quad.) pol./High-sensitive
            # '19': Full (Quad.) pol./Fine
            print('This reader only supports ScanSAR full aperture data parsing.')
            raise Exception('Use stripmap reader for other modes')

        if operationMode != '08':
            raise Exception('Only ScanSAR nominal mode is currently supported')
        
        # Extract the I and Q channels
        imageData = CEOS.CEOSDB(xml=self.image_record,dataFile=fp)

        ###If only a VRT needs to be written
        for line in range(self.length):
            if ((line%1000) == 0):
                self.logger.debug("Extracting line %s" % line)
                imageData.parseFast()

                ###Always read the first line virtual / not
                if line==0:
                    offsetAfterFirstImageRecord = fp.tell()
                    yr = imageData.metadata['Sensor acquisition year']
                    dys = imageData.metadata['Sensor acquisition day of year']
                    msecs = imageData.metadata['Sensor acquisition milliseconds of day']
                    usecs = imageData.metadata['Sensor acquisition micro-seconds of day']
                    self.sensingStart = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds = usecs*1e-6)
                    self.nearRange = imageData.metadata['Slant range to 1st data sample']
                    self.prf = imageData.metadata['PRF'] * 1.0e-3
                    sceneCenterIncidenceAngle = self.parent.leaderFile.sceneHeaderRecord.metadata['Incidence angle at scene centre']
                    sarChannelId = imageData.metadata['SAR channel indicator']
                    scanId = imageData.metadata['Scan ID'] #Scan ID starts with 1

            ###Exit loop after first line if virtual
            if virtual:
                break


            ###Write line to file if not virtual
            IQLine = np.fromfile(fp, dtype='>f', count=2*dataLen)
            self.writeRawData(output, IQLine)


        fp.close()

        ####If virtual file was requested, create VRT here
        if virtual:
            ##Close input file
            with open(output, 'w') as fid:
                fid.write('''<VRTDataset rasterXSize="{0}" rasterYSize="{1}">
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTRawRasterBand">
        <SourceFilename relativeToVRT="0">{2}</SourceFilename>
        <ByteOrder>MSB</ByteOrder>
        <ImageOffset>{3}</ImageOffset>
        <PixelOffset>8</PixelOffset>
        <LineOffset>{4}</LineOffset>
    </VRTRasterBand>
</VRTDataset>'''.format(self.width, self.length,
                       os.path.abspath(self.file),
                       offsetAfterFirstImageRecord,
                       dataLen*8 + offsetAfterFirstImageRecord - offsetAfterImageFDR))

        else:
            ##Close actual file on disk
            output.close()


            
        #burst parameters, currently only for the second, dual polarization, ScanSAR nominal mode 
        #that is the second WBD mode.
        #p.25 and p.115 of ALOS-2/PALSAR-2 Level 1.1/1.5/2.1/3.1 CEOS SAR Product Format Description
    #for the definations of wide swath mode
        nbraw = [358,        470,        358,        355,        487]
        ncraw = [2086.26,    2597.80,    1886.18,    1779.60,    2211.17]

        self.parent.frame.nbraw = nbraw[scanId-1]
        self.parent.frame.ncraw = ncraw[scanId-1]

        #this is the prf fraction (total azimuth bandwith) used in extracting burst.
        #here the total bandwith is 0.93 * prfs[3] for all subswaths, which is the following values:
        #[0.7933, 0.6371, 0.8774, 0.9300, 0.7485] 
        prfs=[2661.847, 3314.512, 2406.568, 2270.575, 2821.225]


        #Only needed for burst extraction. Skipping for now ....
        #self.parent.frame.prffrac = 0.93 * prfs[3]/prfs[scanId-1]


        self.sensingStop = self.sensingStart + datetime.timedelta(seconds = (self.length-1)/self.prf)

    def _calculateRawDimensions(self,fp):
        """
            Run through the data file once, and calculate the valid sampling window start time range.
        """
        self.length = self.imageFDR.metadata['Number of SAR DATA records']
        #self.width = self.imageFDR.metadata['SAR DATA record length']
        self.width = self.imageFDR.metadata['Number of pixels per line per SAR channel']

        return None
