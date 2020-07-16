#!/usr/bin/env python3
import isceobj
import numpy as np
import os
import datetime 
from isceobj.Constants import SPEED_OF_LIGHT
import logging

logger = logging.getLogger('isce.grdsar.topo')

def filenameWithLooks(inname, azlooks, rglooks):
    spl = os.path.splitext(inname)
    ext = '.{0}alks_{1}rlks'.format(azlooks,rglooks)
    outfile = spl[0] + ext + spl[1]
    return outfile


def runTopo(self, method='legendre'):

    from zerodop.topozero import createTopozero
    from isceobj.Planet.Planet import Planet


    refPol = self._grd.polarizations[0]
    reference = self._grd.loadProduct( os.path.join(self._grd.outputFolder,
                                        'beta_{0}.xml'.format(refPol)))


    azlooks, rglooks = self._grd.getLooks(self.posting, reference.azimuthPixelSize,
            reference.groundRangePixelSize, self.numberAzimuthLooks,
            self.numberRangeLooks)


    if (azlooks == 1) and (rglooks == 1):
        rangeName = reference.slantRangeImage.filename

    else:
        rangeName = filenameWithLooks(reference.slantRangeImage.filename,
                        azlooks, rglooks)

    print('Range name : ', rangeName)
    ####Dem name
    demname = self.verifyDEM()
    print('DEM name: ', demname)
    demImg = isceobj.createDemImage()
    demImg.load(demname + '.xml')


    os.makedirs(self._grd.geometryFolder, exist_ok=True)


    #####Run Topo
    planet = Planet(pname='Earth')
    topo = createTopozero()
    topo.prf = 1.0  / reference.azimuthTimeInterval
    topo.radarWavelength = reference.radarWavelength
    topo.orbit = reference.orbit
    topo.width = reference.numberOfSamples // rglooks
    topo.length = reference.numberOfLines // azlooks
    topo.wireInputPort(name='dem', object=demImg)
    topo.wireInputPort(name='planet', object=planet)
    topo.numberRangeLooks = 1
    topo.numberAzimuthLooks = azlooks
    topo.lookSide = reference.side
    topo.sensingStart = reference.sensingStart + datetime.timedelta(seconds = ((azlooks - 1) /2) * reference.azimuthTimeInterval) 
    topo.slantRangeFilename = rangeName

    topo.demInterpolationMethod='BIQUINTIC'
    topo.orbitInterpolationMethod = method.upper()

    topo.latFilename = os.path.join(self._grd.geometryFolder, 'lat.rdr')
    topo.lonFilename = os.path.join(self._grd.geometryFolder, 'lon.rdr')
    topo.heightFilename = os.path.join(self._grd.geometryFolder, 'z.rdr')
    topo.losFilename = os.path.join(self._grd.geometryFolder, self._grd.losFileName)
    topo.incFilename = os.path.join(self._grd.geometryFolder, self._grd.incFileName)
    topo.maskFilename = os.path.join(self._grd.geometryFolder, self._grd.slMaskFileName)
    topo.slantRangeFilename = rangeName

    topo.topo()


    runSimamp(self._grd.geometryFolder)

    return

def runSimamp(outdir, hname='z.rdr'):
    from iscesys.StdOEL.StdOELPy import create_writer
    
    #####Run simamp
    stdWriter = create_writer("log","",True,filename='sim.log')
    objShade = isceobj.createSimamplitude()
    objShade.setStdWriter(stdWriter)


    hgtImage = isceobj.createImage()
    hgtImage.load(os.path.join(outdir, hname) + '.xml')
    hgtImage.setAccessMode('read')
    hgtImage.createImage()

    simImage = isceobj.createImage()
    simImage.setFilename(os.path.join(outdir, 'simamp.rdr'))
    simImage.dataType = 'FLOAT'
    simImage.setAccessMode('write')
    simImage.setWidth(hgtImage.getWidth())
    simImage.createImage()

    objShade.simamplitude(hgtImage, simImage, shade=3.0)

    simImage.renderHdr()
    hgtImage.finalizeImage()
    simImage.finalizeImage()

