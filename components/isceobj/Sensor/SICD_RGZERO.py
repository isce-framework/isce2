#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Piyush Agram
# Copyright 2010, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged. Any commercial
# use must be negotiated with the Office of Technology Transfer at the
# California Institute of Technology.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
#                        NASA Jet Propulsion Laboratory
#                      California Institute of Technology
#                        (C) 2010  All Rights Reserved
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime
import isceobj
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.AstronomicalHandbook import Const
from isceobj.Planet.Planet import Planet
from isceobj.Scene.Frame import Frame
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil
from iscesys.Component.Component import Component
from isceobj.Sensor.Sensor import Sensor


SICD = Component.Parameter(
        'sicd',
        public_name='SICD',
        default=None,
        type=str,
        mandatory=True,
        intent='input',
        doc='SICD input file')

class SICD_RGZERO(Sensor):
    """
        A class to parse SICD RGZERO metadata
    """

    parameter_list = (SICD,) + Sensor.parameter_list
    logging_name = "isce.sensor.SICD_RGZERO"
    family_name = "sicd_rgzero"

    def __init__(self):
        super(SICD_RGZERO,self).__init__()
        self._sicdmeta = None

        return None

    def getFrame(self):
        return self.frame


    def parse(self):
        try:
            import sarpy.io.complex as cf
        except ImportError:
            raise Exception('You need to install sarpy from NGA - https://github.com/ngageoint/sarpy to work with SICD data')
        self._sicdmeta = cf.open(self.sicd).sicdmeta
        self.populateMetadata()


    def _populatePlatform(self):
        mdict = self._sicdmeta
        platform = self.frame.getInstrument().getPlatform()

        platform.setMission(mdict.CollectionInfo.CollectorName)
        platform.setPlanet(Planet(pname="Earth"))
        side = mdict.SCPCOA.SideOfTrack
        if side.startswith('R'):
            side = -1
        else:
            side = 1
        platform.setPointingDirection(side)

        if mdict.CollectionInfo.RadarMode.ModeType.upper() != 'STRIPMAP':
            raise Exception('SICD ModeType should be STRIPMAP')

        if mdict.CollectionInfo.CollectType.upper() != 'MONOSTATIC':
            raise Exception('SICD ModeType should be MONOSTATIC')


    def _populateInstrument(self, mdict=None):
        if mdict is None:
            mdict = self._sicdmeta

        instrument = self.frame.getInstrument()

        ###Ensure that data is actually SICD RGZERO
        if (mdict.Grid.Type != 'RGZERO'):
            raise Exception('Input data must be SICD RGZERO')

        if (mdict.Grid.ImagePlane != 'SLANT'):
            raise Exception('Input data must be SICD RGZERO in Slant Range plane')
        
        rangePixelSize = mdict.Grid.Row.SS
        azimuthPixelSize = mdict.Grid.Col.SS
        fs = Const.c/(2*rangePixelSize)

        fc = mdict.RMA.INCA.FreqZero
        prf = mdict.Timeline.IPP.Set.IPPPoly[1] * mdict.ImageFormation.RcvChanProc.PRFScaleFactor 

        instrument.setRadarWavelength(Const.c/fc)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setRangePixelSize(rangePixelSize)

        try:
            WFParams = mdict.RadarCollection.Waveform.WFParameters[0]
        except TypeError:
            WFParams = mdict.RadarCollection.Waveform.WFParameters

        instrument.setPulseLength(WFParams.TxPulseLength)
        instrument.setChirpSlope(WFParams.TxRFBandwidth / WFParams.TxPulseLength )
        instrument.setRangeSamplingRate(fs)
        instrument.setInPhaseValue(0.)
        instrument.setQuadratureValue(0.)
        instrument.platform.setAntennaLength(2.2 * azimuthPixelSize)

    def _populateFrame(self, mdict=None):
        if mdict is None:
            mdict = self._sicdmeta

        startRange = mdict.RMA.INCA.R_CA_SCP - (mdict.ImageData.SCPPixel.Row * mdict.Grid.Row.SS)

        ####Compute the UTC times
        zd_t_scp = mdict.RMA.INCA.TimeCAPoly[0]
        ss_zd_s = 1 /self.frame.PRF
        sensingStart = mdict.Timeline.CollectStart + datetime.timedelta(seconds = (zd_t_scp - mdict.ImageData.SCPPixel.Col * ss_zd_s))
        sensingStop = sensingStart + datetime.timedelta(seconds = (mdict.ImageData.NumCols-1) / self.frame.PRF)
        sensingMid = sensingStart + 0.5 * (sensingStop - sensingStart)
        
        self.frame.setStartingRange(startRange)
        if mdict.SCPCOA.ARPVel.Z > 0:
            self.frame.setPassDirection('ASCENDING')
        else:
            self.frame.setPassDirection('DESCENDING')

        self.frame.setOrbitNumber(9999)
        self.frame.setProcessingFacility(mdict.ImageCreation.Site)
        self.frame.setProcessingSoftwareVersion(mdict.ImageCreation.Application)

        pol = mdict.ImageFormation.TxRcvPolarizationProc
        self.frame.setPolarization(pol[0] + pol[2])
        self.frame.setNumberOfLines(mdict.ImageData.NumCols)
        self.frame.setNumberOfSamples(mdict.ImageData.NumRows)


        self.frame.setSensingStart(sensingStart)
        self.frame.setSensingMid(sensingMid)
        self.frame.setSensingStop(sensingStop)

        rangePixelSize = self.frame.getInstrument().getRangePixelSize()
        farRange = startRange +  self.frame.getNumberOfSamples()*rangePixelSize
        self.frame.setFarRange(farRange)


    def _populateOrbit(self, mdict=None):
        import numpy.polynomial.polynomial as poly
        if mdict is None:
            mdict = self._sicdmeta
        
        raw_start_time = mdict.Timeline.CollectStart

        tmin = self.frame.sensingStart - datetime.timedelta(seconds=5)
        tmax = self.frame.sensingStop + datetime.timedelta(seconds=5)


        orbit = self.frame.getOrbit()
        orbit.setReferenceFrame('ECEF')
        orbit.setOrbitSource('Header')
       
        posX = mdict.Position.ARPPoly.X 
        posY = mdict.Position.ARPPoly.Y
        posZ = mdict.Position.ARPPoly.Z
        velX = poly.polyder(posX)
        velY = poly.polyder(posY)
        velZ = poly.polyder(posZ)

        tinp = tmin
        while tinp <= tmax:

            deltaT = (tinp - raw_start_time).total_seconds()
            vec = StateVector()
            vec.setTime(tinp)
            vec.setPosition([poly.polyval(deltaT, posX),
                             poly.polyval(deltaT, posY),
                             poly.polyval(deltaT, posZ)])
            vec.setVelocity([poly.polyval(deltaT, velX),
                             poly.polyval(deltaT, velY),
                             poly.polyval(deltaT, velZ)])

            orbit.addStateVector(vec)
            tinp = tinp + datetime.timedelta(seconds=1)


    def populateImage(self):
        import sarpy.io.complex as cf

        img = cf.open(self.sicd)
        data = img.read_chip()
        if self._sicdmeta.SCPCOA.SideOfTrack.startswith('R'):
            viewarr = data
        else:
            viewarr = data[:,::-1]
        
        data.T.tofile(self.output)

        rawImage = isceobj.createSlcImage()
        rawImage.setByteOrder('l')
        rawImage.setFilename(self.output)
        rawImage.setAccessMode('read')
        rawImage.setWidth(self.frame.getNumberOfSamples())
        rawImage.setXmax(self.frame.getNumberOfSamples())
        rawImage.setXmin(0)
        self.getFrame().setImage(rawImage)
        #rawImage.renderHdr()

    def _populateExtras(self):
        """
        Populate some extra fields.
        """
        from sarpy.geometry.point_projection import coa_projection_set
        import numpy as np
        mdict = self._sicdmeta

        ###Imagesize
        rows = np.linspace(0., mdict.ImageData.NumRows*1.0, num=3)
        rdot = []

        for grow in rows:
            pt = coa_projection_set(mdict,[grow,0])
            rdot.append( pt[1][0])

        self.frame._dopplerVsPixel = list(np.polyfit(rows, rdot, 2)[::-1])


    def extractImage(self):
        """Extract the raw image data"""
        self.parse()
        self._populateExtras()
        self.populateImage()

    def extractDoppler(self):
        """
        Return the doppler centroid as defined in the HDF5 file.
        """
        dopp = self.frame._dopplerVsPixel
        quadratic = {}
        quadratic['a'] = dopp[0]
        quadratic['b'] = dopp[1]
        quadratic['c'] = dopp[2]
        return quadratic

