#!/usr/bin/env python3

#Author: Cunren Liang, 2015-

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

import isceobj

#changed to use the following parameters
IMAGE_FILE = Component.Parameter('imageFile',
        public_name='image file',
        type = str,
        default=None,
        mandatory = True,
        doc = 'ALOS-2 CEOS image file')

LEADER_FILE = Component.Parameter('leaderFile',
        public_name='leader file',
        type = str,
        default=None,
        mandatory = True,
        doc = 'ALOS-2 CEOS leader file')

OUTPUT_FILE = Component.Parameter('outputFile',
        public_name='output file',
        type = str,
        default=None,
        mandatory = True,
        doc = 'output file')

USE_VIRTUAL_FILE = Component.Parameter('useVirtualFile',
        public_name='use virtual file',
        type=bool,
        default=True,
        mandatory=False,
        doc='use virtual files instead of using disk space')

####List of facilities
TRACK = Component.Facility('track',
        public_name='track',
        module = 'isceobj.Sensor.MultiMode',
        factory='createTrack',
        args = (),
        mandatory = True,
        doc = 'A track of ALOS-2 SLCs populated by the reader')


class ALOS2(Component):
    """
    ALOS-2 multi-mode reader
    """
    family = 'alos2multimode'
    logging = 'isce.sensor.alos2multimode'

    parameter_list = (IMAGE_FILE,
                      LEADER_FILE,
                      OUTPUT_FILE,
                      USE_VIRTUAL_FILE)

    facility_list = (TRACK,)

    # Range sampling rate
    fsampConst = { 104: 1.047915957140240E+08,
                   52: 5.239579785701190E+07,
                   34: 3.493053190467460E+07,
                   17: 1.746526595233730E+07 }
    # Orbital Elements (Quality) Designator, data format P68
    orbitElementsDesignator = {'0':'0: preliminary',
                               '1':'1: decision',
                               '2':'2: high precision'}
    # Operation mode, data format P50
    operationModeDesignator = {'00': '00: Spotlight mode',
                               '01': '01: Ultra-fine mode',
                               '02': '02: High-sensitive mode',
                               '03': '03: Fine mode',
                               '04': '04: spare',
                               '05': '05: spare',
                               '08': '08: ScanSAR nominal mode',
                               '09': '09: ScanSAR wide mode',
                               '18': '18: Full (Quad.) pol./High-sensitive mode',
                               '19': '19: Full (Quad.) pol./Fine mode',
                               '64': '64: Manual observation'}

    def __init__(self, name=''):
        super().__init__(family=self.__class__.family, name=name) 

        return


    def readImage(self):
        '''
        read image and get parameters
        '''
        try:
            fp = open(self.imageFile,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return

        #read meta data
        imageFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_file.xml'), dataFile=fp)
        imageFDR.parse()
        fp.seek(imageFDR.getEndOfRecordPosition())

        #record length: header (544 bytes) + SAR data (width*8 bytes)
        recordlength = imageFDR.metadata['SAR DATA record length']
        width =  imageFDR.metadata['Number of pixels per line per SAR channel']
        length = imageFDR.metadata['Number of SAR DATA records']

        #line header
        imageData = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/image_record.xml'), dataFile=fp)
        imageData.parseFast()

        #creat vrt and xml files for sar data
        image = isceobj.createSlcImage()
        image.setFilename(self.outputFile)
        image.setWidth(width)
        image.setLength(length)
        image.renderHdr()

        #sar data
        fileHeaderBytes = 720
        lineHeaderBytes = 544
        if self.useVirtualFile:
            #this overwrites the previous vrt
            with open(self.outputFile+'.vrt', 'w') as fid:
                fid.write('''<VRTDataset rasterXSize="{0}" rasterYSize="{1}">
    <VRTRasterBand band="1" dataType="CFloat32" subClass="VRTRawRasterBand">
        <SourceFilename relativeToVRT="0">{2}</SourceFilename>
        <ByteOrder>MSB</ByteOrder>
        <ImageOffset>{3}</ImageOffset>
        <PixelOffset>8</PixelOffset>
        <LineOffset>{4}</LineOffset>
    </VRTRasterBand>
</VRTDataset>'''.format(width, length,
                       self.imageFile,
                       fileHeaderBytes + lineHeaderBytes,
                       width*8 + lineHeaderBytes))
        else:
            #read sar data line by line
            try:
                fp2 = open(self.outputFile,'wb')
            except IOError as errs:
                errno,strerr = errs
                print("IOError: %s" % strerr)
                return
            fp.seek(-lineHeaderBytes, 1)
            for line in range(length):
                if (((line+1)%1000) == 0):
                    print("extracting line %6d of %6d" % (line+1, length), end='\r', flush=True)
                fp.seek(lineHeaderBytes, 1)
                IQLine = np.fromfile(fp, dtype='>f', count=2*width)
                self.writeRawData(fp2, IQLine)
                #IQLine.tofile(fp2)
            print("extracting line %6d of %6d" % (length, length))
            fp2.close()

        #close input image file
        fp.close()

        return (imageFDR, imageData)


    def readLeader(self):
        '''
        read meta data from leader
        '''
        try:
            fp = open(self.leaderFile,'rb')
        except IOError as errs:
            errno,strerr = errs
            print("IOError: %s" % strerr)
            return
        
        # Leader record
        leaderFDR = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/leader_file.xml'),dataFile=fp)
        leaderFDR.parse()
        fp.seek(leaderFDR.getEndOfRecordPosition())

        # Scene Header
        sceneHeaderRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/scene_record.xml'),dataFile=fp)
        sceneHeaderRecord.parse()
        fp.seek(sceneHeaderRecord.getEndOfRecordPosition())

        # Platform Position
        platformPositionRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/platform_position_record.xml'),dataFile=fp)
        platformPositionRecord.parse()
        fp.seek(platformPositionRecord.getEndOfRecordPosition())

        #####Skip attitude information
        fp.seek(16384,1)
        #####Skip radiometric information
        fp.seek(9860,1)
        ####Skip the data quality information
        fp.seek(1620,1)
        ####Skip facility 1-4
        fp.seek(325000 + 511000 + 3072 + 728000, 1)

        ####Read facility 5
        facilityRecord = CEOS.CEOSDB(xml=os.path.join(xmlPrefix,'alos2_slc/facility_record.xml'), dataFile=fp)
        facilityRecord.parse()

        ###close file
        fp.close()

        return (leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord)


    def setTrack(self, leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData):
        '''
        set track parameters
        '''
        track = self.track

        #passDirection
        passDirection = sceneHeaderRecord.metadata['Time direction indicator along line direction']
        if passDirection == 'ASCEND':
            track.passDirection = 'ascending'
        elif passDirection == 'DESCEND':
            track.passDirection = 'descending'
        else:
            raise Exception('Unknown passDirection. passDirection = {0}'.format(passDirection))

        #pointingDirection
        ######ALOS-2 includes this information in clock angle
        clockAngle = sceneHeaderRecord.metadata['Sensor clock angle']
        if clockAngle == 90.0:
            track.pointingDirection = 'right'
        elif clockAngle == -90.0:
            track.pointingDirection = 'left'
        else:
            raise Exception('Unknown look side. Clock Angle = {0}'.format(clockAngle))

        #operation mode
        track.operationMode = self.operationModeDesignator[
                              (sceneHeaderRecord.metadata['Sensor ID and mode of operation for this channel'])[10:12]
                              ]

        #use this instead. 30-JAN-2020
        track.operationMode = os.path.basename(self.leaderFile).split('-')[-1][0:3]

        #radarWavelength
        track.radarWavelength = sceneHeaderRecord.metadata['Radar wavelength']

        #orbit
        orb = self.readOrbit(platformPositionRecord)
        track.orbit.setOrbitSource(orb.getOrbitSource())
        track.orbit.setOrbitQuality(orb.getOrbitQuality())
        #add orbit from frame
        for sv in orb:
            addOrbit = True
            #Orbit class does not check this
            for x in track.orbit:
                if x.getTime() == sv.getTime():
                    addOrbit = False
                    break
            if addOrbit:
                track.orbit.addStateVector(sv)

        # the following are to be set when mosaicking frames.
        # 'numberOfSamples',
        # 'numberOfLines',
        # 'startingRange',
        # 'rangeSamplingRate',
        # 'rangePixelSize',
        # 'sensingStart',
        # 'prf',
        # 'azimuthPixelSize'


    def setFrame(self, leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData):
        '''
        set frame parameters
        '''
        frame = self.track.frames[-1]

        #get frame number from file name
        frame.frameNumber = os.path.basename(self.imageFile).split('-')[2][-4:]
        frame.processingFacility = sceneHeaderRecord.metadata['Processing facility identifier']
        frame.processingSystem = sceneHeaderRecord.metadata['Processing system identifier']
        frame.processingSoftwareVersion = sceneHeaderRecord.metadata['Processing version identifier']
        #orbit quality
        orb = self.readOrbit(platformPositionRecord)
        frame.orbitQuality = orb.getOrbitQuality()

        # the following are to be set when mosaicking swaths
        # 'numberOfSamples',
        # 'numberOfLines',
        # 'startingRange',
        # 'rangeSamplingRate',
        # 'rangePixelSize',
        # 'sensingStart',
        # 'prf',
        # 'azimuthPixelSize'


    def setSwath(self, leaderFDR, sceneHeaderRecord, platformPositionRecord, facilityRecord, imageFDR, imageData):
        '''
        set swath parameters
        '''
        swath = self.track.frames[-1].swaths[-1]
        operationMode = (sceneHeaderRecord.metadata['Sensor ID and mode of operation for this channel'])[10:12]
        
        #set swath number here regardless of operation mode, will be updated for ScanSAR later
        swath.swathNumber = 1

        #polarization
        polDesignator = {0: 'H',
                         1: 'V'}
        swath.polarization = '{}{}'.format(polDesignator[imageData.metadata['Transmitted polarization']],
                             polDesignator[imageData.metadata['Received polarization']])

        #image dimensions
        swath.numberOfSamples = imageFDR.metadata['Number of pixels per line per SAR channel']
        swath.numberOfLines = imageFDR.metadata['Number of SAR DATA records']

        #range
        swath.startingRange = imageData.metadata['Slant range to 1st data sample']
        swath.rangeSamplingRate = self.fsampConst[int(sceneHeaderRecord.metadata['Range sampling rate in MHz'])]
        swath.rangePixelSize = Const.c/(2.0*swath.rangeSamplingRate)
        swath.rangeBandwidth =abs((sceneHeaderRecord.metadata['Nominal range pulse (chirp) amplitude coefficient linear term']) *
                               (sceneHeaderRecord.metadata['Range pulse length in microsec']*1.0e-6))

        #sensingStart
        yr = imageData.metadata['Sensor acquisition year']
        dys = imageData.metadata['Sensor acquisition day of year']
        msecs = imageData.metadata['Sensor acquisition milliseconds of day']
        usecs = imageData.metadata['Sensor acquisition micro-seconds of day']
        swath.sensingStart = datetime.datetime(yr,1,1) + datetime.timedelta(days=(dys-1)) + datetime.timedelta(seconds = usecs*1e-6)

        #prf
        if operationMode == '08' or operationMode == '09':
            # Operation mode
            # '00': 'Spotlight mode',
            # '01': 'Ultra-fine mode',
            # '02': 'High-sensitive mode',
            # '03': 'Fine mode',
            # '04': 'spare',
            # '05': 'spare',
            # '08': 'ScanSAR nominal mode',
            # '09': 'ScanSAR wide mode',
            # '18': 'Full (Quad.) pol./High-sensitive mode',
            # '19': 'Full (Quad.) pol./Fine mode',
            # '64': 'Manual observation'
            #print('ScanSAR mode, using PRF from the line header')
            swath.prf = imageData.metadata['PRF'] * 1.0e-3
        else:
            #print('not ScanSAR mode, using PRF from leader file')
            swath.prf = sceneHeaderRecord.metadata['Pulse Repetition Frequency in mHz']*1.0e-3

        #azimuth pixel size at swath center on ground
        azimuthTime = swath.sensingStart + datetime.timedelta(seconds=swath.numberOfLines/swath.prf/2.0)
        orbit = self.readOrbit(platformPositionRecord)
        svmid = orbit.interpolateOrbit(azimuthTime, method='hermite')
        height = np.linalg.norm(svmid.getPosition())
        velocity = np.linalg.norm(svmid.getVelocity())
        #earth radius in meters
        r = 6371 * 1000.0
        swath.azimuthPixelSize = velocity / swath.prf * r / height
        swath.azimuthLineInterval = 1.0 / swath.prf

        #doppler
        swath.dopplerVsPixel = self.reformatDoppler(sceneHeaderRecord, imageFDR, imageData)

        #azimuth FM rate
        azimuthTime = swath.sensingStart + datetime.timedelta(seconds=swath.numberOfLines/swath.prf/2.0)
        swath.azimuthFmrateVsPixel = self.computeAzimuthFmrate(sceneHeaderRecord, platformPositionRecord, imageFDR, imageData, azimuthTime)


        #burst information estimated from azimuth spectrum. Cunren, 14-DEC-2015
        if operationMode == '08' or operationMode == '09':
            sceneCenterIncidenceAngle = sceneHeaderRecord.metadata['Incidence angle at scene centre']
            sarChannelId = imageData.metadata['SAR channel indicator']
            #Scan ID starts with 1, ScanSAR = 1 to 7, Except ScanSAR = 0
            scanId = imageData.metadata['Scan ID']
            swath.swathNumber = scanId

            #looks like all ScanSAR nominal modes (WBS,WBD,WWS,WWD) have same burst parameters, so remove the limitation here
            #if (sceneCenterIncidenceAngle > 39.032 - 5.0 and sceneCenterIncidenceAngle < 39.032 + 5.0) and (sarChannelId == 2):
            if operationMode == '08':
                #burst parameters, currently only for the second, dual polarization, ScanSAR nominal mode
                #that is the second WBD mode
                #p.25 and p.115 of ALOS-2/PALSAR-2 Level 1.1/1.5/2.1/3.1 CEOS SAR Product Format Description
                #for the definations of wide swath mode
                nbraw = [358,        470,        358,        355,        487]
                ncraw = [2086.26,    2597.80,    1886.18,    1779.60,    2211.17]

                swath.burstLength = nbraw[scanId-1]
                swath.burstCycleLength = ncraw[scanId-1]

                #this is the prf fraction (total azimuth bandwith) used in extracting burst
                #here the total bandwith is 0.93 * prfs[3] for all subswaths, which is the following values:
                #[0.7933, 0.6371, 0.8774, 0.9300, 0.7485] 
                prfs=[2661.847, 3314.512, 2406.568, 2270.575, 2821.225]
                swath.prfFraction = 0.93 * prfs[3]/prfs[scanId-1]

        #compute burst start time
        if operationMode == '08':
            (burstStartTime, burstCycleLengthNew) = self.burstTimeRefining(self.outputFile, 
                swath.numberOfSamples, 
                swath.numberOfLines, 
                swath.prf, 
                swath.burstLength, 
                swath.burstCycleLength, 
                swath.azimuthFmrateVsPixel, 
                swath.sensingStart, 
                self.useVirtualFile)
            swath.burstStartTime = burstStartTime
            swath.burstCycleLength = burstCycleLengthNew


    def computeAzimuthFmrate(self, sceneHeaderRecord, platformPositionRecord, imageFDR, imageData, azimuthTime):
        import copy
        '''
        compute azimuth FM rate, copied from Piyush's code.
        azimuthTime: middle of the scene should be a good time
        '''
        #parameters required
        orbit = self.readOrbit(platformPositionRecord)
        dopplerVsPixel = self.reformatDoppler(sceneHeaderRecord, imageFDR, imageData)
        width = imageFDR.metadata['Number of pixels per line per SAR channel']

        startingRange = imageData.metadata['Slant range to 1st data sample']
        rangeSamplingRate = self.fsampConst[int(sceneHeaderRecord.metadata['Range sampling rate in MHz'])]
        radarWavelength = sceneHeaderRecord.metadata['Radar wavelength']

        clockAngle = sceneHeaderRecord.metadata['Sensor clock angle']
        if clockAngle == 90.0:
            #right
            pointingDirection = -1
        elif clockAngle == -90.0:
            #left
            pointingDirection = 1
        else:
            raise Exception('Unknown look side. Clock Angle = {0}'.format(clockAngle))

        ##We have to compute FM rate here.
        ##Cunren's observation that this is all set to zero in CEOS file.
        ##Simplification from Cunren's fmrate.py script
        ##Should be the same as the one in focus.py
        planet = Planet(pname='Earth')
        elp = copy.copy(planet.ellipsoid)
        svmid = orbit.interpolateOrbit(azimuthTime, method='hermite')
        xyz = svmid.getPosition()
        vxyz = svmid.getVelocity()
        llh = elp.xyz_to_llh(xyz)
        hdg = orbit.getENUHeading(azimuthTime)

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

        fmrate = []
        lookSide = pointingDirection
        centerVelNorm = np.linalg.norm(centerVel)

        ##Retaining Cunren's code for computing at every pixel.
        ##Can be done every 10th pixel since we only fit a quadratic/ cubic.
        ##Also can be vectorized for speed.

        for ii in range(width):
            rg = startingRange + ii * 0.5 * Const.c / rangeSamplingRate
            #don't forget to flip coefficients
            dop = np.polyval(dopplerVsPixel[::-1], ii)

            th = np.arccos(((avghgt+radiusOfCurvature)**2 + rg**2 -radiusOfCurvature**2)/(2.0 * (avghgt + radiusOfCurvature) * rg))
            thaz = np.arcsin(((radarWavelength*dop/(2.0*np.sin(th))) + (centerVel[2] / np.tan(th))) / np.sqrt(centerVel[0]**2 + centerVel[1]**2)) - lookSide * np.arctan(centerVel[1]/centerVel[0])

            lookVec = [ np.sin(th) * np.sin(thaz),
                        np.sin(th) * np.cos(thaz) * lookSide,
                        -np.cos(th)]

            vdotl = np.dot(lookVec, centerVel)
            adotl = np.dot(lookVec, centerAcc)
            fmratex = 2.0*(adotl + (vdotl**2 - centerVelNorm**2)/rg)/(radarWavelength)
            fmrate.append(fmratex)

        ##Fitting order 2 polynomial to FM rate
        p = np.polyfit(np.arange(width), fmrate, 2)
        azimuthFmrateVsPixel = [p[2], p[1], p[0], 0.]

        return azimuthFmrateVsPixel


    def reformatDoppler(self, sceneHeaderRecord, imageFDR, imageData):
        '''
        reformat Doppler coefficients
        '''
        dopplerCoeff = [sceneHeaderRecord.metadata['Doppler center frequency constant term'],
        sceneHeaderRecord.metadata['Doppler center frequency linear term']]

        width = imageFDR.metadata['Number of pixels per line per SAR channel']
        startingRange = imageData.metadata['Slant range to 1st data sample']
        rangeSamplingRate = self.fsampConst[int(sceneHeaderRecord.metadata['Range sampling rate in MHz'])]
        
        rng = startingRange + np.arange(0,width,100) * 0.5 * Const.c / rangeSamplingRate
        doppler = dopplerCoeff[0] + dopplerCoeff[1] * rng / 1000.
        dfit = np.polyfit(np.arange(0, width, 100), doppler, 1)
        dopplerVsPixel = [dfit[1], dfit[0], 0., 0.]

        return dopplerVsPixel


    def readOrbit(self, platformPositionRecord):
        '''
        reformat orbit from platformPositionRecord
        '''
        orb=Orbit()
        orb.setOrbitSource('leaderfile')
        orb.setOrbitQuality(self.orbitElementsDesignator[platformPositionRecord.metadata['Orbital elements designator']])

        t0 = datetime.datetime(year=platformPositionRecord.metadata['Year of data point'],
                               month=platformPositionRecord.metadata['Month of data point'],
                               day=platformPositionRecord.metadata['Day of data point'])
        t0 = t0 + datetime.timedelta(seconds=platformPositionRecord.metadata['Seconds of day'])

        #####Read in orbit in inertial coordinates
        deltaT = platformPositionRecord.metadata['Time interval between data points']
        numPts = platformPositionRecord.metadata['Number of data points']
        for i in range(numPts):
            vec = StateVector()
            t = t0 + datetime.timedelta(seconds=i*deltaT)
            vec.setTime(t)

            dataPoints = platformPositionRecord.metadata['Positional Data Points'][i]
            pos = [dataPoints['Position vector X'], dataPoints['Position vector Y'], dataPoints['Position vector Z']]
            vel = [dataPoints['Velocity vector X'], dataPoints['Velocity vector Y'], dataPoints['Velocity vector Z']]
            vec.setPosition(pos)
            vec.setVelocity(vel)
            orb.addStateVector(vec)

        return orb


    def burstTimeRefining(self, slcFile, numberOfSamples, numberOfLines, pulseRepetitionFrequency, burstLength, burstCycleLength, azimuthFmrateVsPixel, sensingStart, useVirtualFile=True):
        '''
        compute start time of raw burst
        '''
        #number of lines from start and end of file
        #this mainly considers ALOS-2 full-aperture length, should be updated for ALOS-4?
        delta_line = 15000

        #first estimate at file start
        start_line1 = delta_line
        (burstStartLine1, burstStartTime1, burstStartLineEstimated1) = self.burstTime(slcFile, 
            numberOfSamples, 
            numberOfLines, 
            pulseRepetitionFrequency, 
            burstLength, 
            burstCycleLength, 
            azimuthFmrateVsPixel, 
            sensingStart, 
            start_line1, 
            1000, 
            1, 
            useVirtualFile)

        #estimate again at file end
        #number of burst cycles
        num_nc = np.around((numberOfLines - delta_line*2) / burstCycleLength)
        start_line2 = int(np.around(start_line1 + num_nc * burstCycleLength))
        (burstStartLine2, burstStartTime2, burstStartLineEstimated2) = self.burstTime(slcFile, 
            numberOfSamples, 
            numberOfLines, 
            pulseRepetitionFrequency, 
            burstLength, 
            burstCycleLength, 
            azimuthFmrateVsPixel, 
            sensingStart, 
            start_line2, 
            1000, 
            1, 
            useVirtualFile)

        #correct burst cycle value
        LineDiffIndex = 0
        LineDiffMin = np.fabs(burstStartLineEstimated1 + burstCycleLength * LineDiffIndex - burstStartLineEstimated2)
        for i in range(0, 100000):
            LineDiffMinx = np.fabs(burstStartLineEstimated1 + burstCycleLength * i - burstStartLineEstimated2)
            if LineDiffMinx <= LineDiffMin:
                LineDiffMin = LineDiffMinx
                LineDiffIndex = i
        burstCycleLengthNew = burstCycleLength - (burstStartLineEstimated1 + burstCycleLength * LineDiffIndex - burstStartLineEstimated2) / LineDiffIndex

        #use correct burstCycleLength to do final estimation
        start_line = int(np.around(numberOfLines/2.0))
        (burstStartLine, burstStartTime, burstStartLineEstimated) = self.burstTime(slcFile, 
            numberOfSamples, 
            numberOfLines, 
            pulseRepetitionFrequency, 
            burstLength, 
            burstCycleLengthNew, 
            azimuthFmrateVsPixel, 
            sensingStart, 
            start_line, 
            1000, 
            1, 
            useVirtualFile)
        
        #return burstStartTime and refined burstCycleLength
        return (burstStartTime, burstCycleLengthNew)


    def burstTime(self, slcFile, numberOfSamples, numberOfLines, pulseRepetitionFrequency, burstLength, burstCycleLength, azimuthFmrateVsPixel, sensingStart, startLine=500, startColumn=500, pow2=1, useVirtualFile=True):
        '''
        compute start time of raw burst
        '''
        #######################################################
        #set these parameters
        width         = numberOfSamples
        length        = numberOfLines
        prf           = pulseRepetitionFrequency
        nb            = burstLength
        nc            = burstCycleLength
        fmrateCoeff   = azimuthFmrateVsPixel
        sensing_start = sensingStart
        saz           = startLine #start line to be used (included, index start with 0. default: 500)
        srg           = startColumn #start column to be used (included, index start with 0. default: 500)
        p2            = pow2 #must be 1(default) or power of 2. azimuth fft length = THIS ARGUMENT * next of power of 2 of full-aperture length.
        #######################################################  

        def create_lfm(ns, it, offset, k):
            '''
            # create linear FM signal
            # ns: number of samples
            # it: time interval of the samples
            # offset: offset
            # k: linear FM rate
            #offset-centered, this applies to both odd and even cases
            '''
            ht = (ns - 1) / 2.0
            t = np.arange(-ht, ht+1.0, 1)
            t = (t + offset) * it
            cj = np.complex64(1j)
            lfm = np.exp(cj * np.pi * k * t**2)

            return lfm


        def next_pow2(a):
            x=2
            while x < a:
                x *= 2
            return x


        def is_power2(num):
            '''states if a number is a power of two'''
            return num != 0 and ((num & (num - 1)) == 0)


        if not (p2 == 1 or is_power2(p2)):
            raise Exception('pow2 must be 1 or power of 2\n')

        #fmrate, use the convention that ka > 0
        ka = -np.polyval(fmrateCoeff[::-1], np.arange(width))

        #area to be used for estimation
        naz = int(np.round(nc))            #number of lines to be used.
        eaz = saz+naz-1                    #ending line to be used (included)
        caz = int(np.round((saz+eaz)/2.0)) #central line of the lines used.
        caz_deramp = (saz+eaz)/2.0         #center of deramp signal (may be fractional line number)

        nrg = 400                          #number of columns to be used
        erg = srg+nrg-1                    #ending column to be used (included)
        crg = int(np.round((srg+erg)/2.0)) #central column of the columns used.

        #check parameters
        if not (saz >=0 and saz <= length - 1):
            raise Exception('wrong starting line\n')
        if not (eaz >=0 and eaz <= length - 1):
            raise Exception('wrong ending line\n')
        if not (srg >=0 and srg <= width - 1):
            raise Exception('wrong starting column\n')
        if not (erg >=0 and erg <= width - 1):
            raise Exception('wrong ending column\n')

        #number of lines of a full-aperture
        nfa = int(np.round(prf / ka[crg] / (1.0 / prf)))
        #use nfa to determine fft size. fft size can be larger than this
        nazfft = next_pow2(nfa) * p2

        #deramp signal
        deramp = np.zeros((naz, nrg), dtype=np.complex64)
        for i in range(nrg):
            deramp[:, i] = create_lfm(naz, 1.0/prf, 0, -ka[i+srg])

        #read data, python should be faster
        useGdal = False
        if useGdal:
            from osgeo import gdal
            ###Read in chunk of data
            ds = gdal.Open(slcFile + '.vrt', gdal.GA_ReadOnly)
            data = ds.ReadAsArray(srg, saz, nrg, naz)
            ds = None
        else:
            #!!!hard-coded: ALOS-2 image file header 720 bytes, line header 544 bytes
            if useVirtualFile == True:
                fileHeader = 720
                lineHeader = 544
                #lineHeader is just integer multiples of complex pixle size, so it's good
                lineHeaderSamples = int(lineHeader/np.dtype(np.complex64).itemsize)
                slcFile = self.find_keyword(slcFile+'.vrt', 'SourceFilename')
            else:
                fileHeader = 0
                lineHeader = 0
                lineHeaderSamples = 0
            data = np.memmap(slcFile, np.complex64,'r', offset = fileHeader, shape=(numberOfLines,lineHeaderSamples+numberOfSamples))
            data = data[saz:eaz+1, lineHeaderSamples+srg:lineHeaderSamples+erg+1]
            if useVirtualFile == True:
                data = data.byteswap()

        #deramp data
        datadr = deramp * data

        #spectrum
        spec = np.fft.fft(datadr, n=nazfft, axis=0)

        #shift zero-frequency component to center of spectrum
        spec = np.fft.fftshift(spec, axes=0)

        specm=np.mean(np.absolute(spec), axis=1)

        #number of samples of the burst in frequncy domain
        nbs  = int(np.round(nb*(1.0/prf)*ka[crg]/prf*nazfft));
        #number of samples of the burst cycle in frequncy domain
        ncs  = int(np.round(nc*(1.0/prf)*ka[crg]/prf*nazfft));
        rect = np.ones(nbs, dtype=np.float32)

        #make sure orders of specm and rect are correct, so that peaks
        #happen in the same order as their corresponding bursts
        corr=np.correlate(specm, rect,'same')

        #find burst spectrum center
        ncs_rh = int(np.round((nazfft - ncs) / 2.0))
        #corr_bc = corr[ncs_rh: ncs_rh+ncs]
        #offset between spectrum center and center
        offset_spec = np.argmax(corr[ncs_rh: ncs_rh+ncs])+ncs_rh - (nazfft - 1.0) / 2.0
        #offset in number of azimuth lines
        offset_naz  = offset_spec / nazfft * prf / ka[crg] / (1.0/prf)

        #start line of burst (fractional line number)
        saz_burst = -offset_naz + caz_deramp - (nb - 1.0) / 2.0

        #find out the start line of all bursts (fractional line number,
        #line index start with 0, line 0 is the first SLC line)
        #now only find first burst
        for i in range(-100000, 100000):
            saz_burstx = saz_burst + nc * i
            st_burstx = sensing_start + datetime.timedelta(seconds=saz_burstx * (1.0/prf))
            if saz_burstx >= 0.0 and saz_burstx <= length:
                burstStartLine = saz_burstx
                burstStartTime = st_burstx
                break
        burstStartLineEstimated = saz_burst

        #dump spectrum and correlation
        debug = False
        if debug:
            specm_corr = ''
            for i in range(nazfft):
                specm_corr += '{:6d}     {:f}     {:6d}     {:f}\n'.format(i, specm[i], i, corr[i])
            specm_corr_name = str(sensingStart.year)[2:] + '%02d' % sensingStart.month + '%02d' % sensingStart.day + '_spec_corr.txt'
            with open(specm_corr_name, 'w') as f:
                f.write(specm_corr)

        return (burstStartLine, burstStartTime, burstStartLineEstimated)


    def find_keyword(self, xmlfile, keyword):
        from xml.etree.ElementTree import ElementTree

        value = None
        xmlx = ElementTree(file=open(xmlfile,'r')).getroot()
        #try 10 times
        for i in range(10):
            path=''
            for j in range(i):
                path += '*/'
            value0 = xmlx.find(path+keyword)
            if value0 != None:
                value = value0.text
                break

        return value


    def writeRawData(self, fp, line):
        '''
        Convert complex integer to complex64 format.
        '''
        cJ = np.complex64(1j)
        data = line[0::2] + cJ * line[1::2]
        data.tofile(fp)



if __name__ == '__main__':

    main()

