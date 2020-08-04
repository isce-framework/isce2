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
# Author: Giangi Sacco
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



from __future__ import print_function
import time
import os

import isce
import isceobj
import iscesys
from iscesys.Compatibility import Compatibility
from isceobj.Pause import pause


from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
from iscesys.Component.Application import Application
from isce.applications.insarApp import _InsarBase, logger


class Dpm(_InsarBase):
    """Dpm Application class:

    Implements Dpm processing flow for a pair of scenes from
    sensor raw data to geocoded correlation.
    """
    def __init__(self):
        super(Dpm, self).__init__()
        ## This indicates something has gone wrong, I must delete geocode.        
        del self.runGeocode

    ## extends _InsarBase_steps, but not in the same was as main
    def _steps(self):
        super(Dpm, self)._steps()

        # Geocode
        self.step('geocodecorifg', func=self.geocodeCorIfg)
        self.step('geocodecor4rlks', func=self.geocodeCor4rlks)
        
        self.step('renderProcDoc', func=self.renderProcDoc)

        self.step('timerend', local='timeEnd',func=time.time)

        self.step('logtime', func=logger.info,
                  delayed_args = (" 'Total Time: %i seconds'%(timeEnd-timeStart)",)
                  )
        return None
    
    def renderProcDoc(self):
        self.insar.procDoc.renderXml()

    def coherence(self):
      self.runCoherence(self.correlation_method)
      self.renderProcDoc()


    def geocodeCorIfg(self):        
        corFilename = self.insar.coherenceFilename
        corintFilename = corFilename.replace('.cor','.corint')
        widthInt = self.insar.resampIntImage.width
        rmg_to_cmplx(corFilename,corintFilename,widthInt)
        corintGeocodeFilename = corintFilename+'.geo'
        demGeocodeFilename = corintFilename+'.demcrop'
        geo = self.runGeocode(corintFilename,
                              widthInt,
                              corintGeocodeFilename,
                              demGeocodeFilename)
        geoWidth = geo.computeGeoImageWidth()
        print("geocodecor: widthGeo = ", geoWidth)
        ifgGeocodeFilename = self.insar.geocodeFilename
        demCropFilename = self.insar.demCropFilename
        topoflatIntFilename = self.insar.topophaseFlatFilename
        widthInt = self.insar.resampIntImage.width
        geo = self.runGeocode(topoflatIntFilename,
                              widthInt,
                              ifgGeocodeFilename,
                              demCropFilename)
        geoWidth = geo.computeGeoImageWidth()
        print("geocodeifg: widthGeo = ", geoWidth)
        
        corGeocodeFilename = corFilename + '.geo'
        cmplx_to_rmg(corintGeocodeFilename, corGeocodeFilename, geoWidth)
        
        self.geo_to_rsc(ifgGeocodeFilename, corGeocodeFilename)
        
        return None

    def geocodeCor4rlks(self):        

        corFilename = self.insar.coherenceFilename
        corintFilename = corFilename.replace('.cor','.corint')
        widthInt = self.insar.resampIntImage.width

        #Not the right place for this block.  Create the 4rlks correlation file and
        # geocode it.
        #~/Util/isce/components/mroipac/looks/Nbymhgt.py
        #                     topophase.cor topophase_4rlks.cor 9480 4 4
        cor4rlksFilename = corFilename.replace('.cor','_4rlks.cor')
        from mroipac.looks.Nbymhgt import Nbymhgt
        nbymh = Nbymhgt()
        nbymh.inputImage = corFilename
        nbymh.outputImage = cor4rlksFilename
        nbymh.width = widthInt
        nbymh.rangeLook = 4
        nbymh.azimuthLook = 4
        nbymh.nbymhgt()
        width4rlksInt = widthInt/4
        corint4rlksFilename = cor4rlksFilename.replace('.cor','.corint')
        rmg_to_cmplx(cor4rlksFilename, corint4rlksFilename, width4rlksInt)
        corint4rlksGeocodeFilename = corint4rlksFilename+'.geo'
        dem4rlksGeocodeFilename = corint4rlksFilename+'.demcrop'
        geo4rlks = self.runGeocode4rlks(corint4rlksFilename,
                                        width4rlksInt,
                                        corint4rlksGeocodeFilename,
                                        dem4rlksGeocodeFilename)
        geo4rlksWidth = geo4rlks.computeGeoImageWidth()
        print("geocodecor: widthGeo = ", geo4rlksWidth)
        
        cor4rlksGeocodeFilename = cor4rlksFilename + '.geo'
        cmplx_to_rmg(corint4rlksGeocodeFilename, cor4rlksGeocodeFilename, geo4rlksWidth)

#        self.geo_to_rsc(ifgGeocodeFilename,corGeocodeFilename)

        return None
    

    def geo_to_rsc(self, ifgGeoFile, corGeoFile):
        from isceobj.XmlUtil.XmlUtil import XmlUtil
        
        xmlDat = {'Coordinate1':{'size':None,'startingValue':None,'delta':None},
                  'Coordinate2':{'size':None,'startingValue':None,'delta':None}}

        rscXML = {'WIDTH':('Coordinate1','size'),
                  'X_FIRST':('Coordinate1','startingValue'),
                  'X_STEP':('Coordinate1','delta'),
                  'FILE_LENGTH':('Coordinate2','size'),
                  'Y_FIRST':('Coordinate2','startingValue'),
                  'Y_STEP':('Coordinate2','delta')}
            
        rscOrder = ('WIDTH','FILE_LENGTH','X_FIRST','X_STEP','Y_FIRST','Y_STEP')
        
        ifgGeoXmlFile = ifgGeoFile + '.xml'

        xu = XmlUtil()
        xuf = xu.readFile(ifgGeoXmlFile)
        c = xuf.findall('component')

        for cx in c:
            cxn = cx.attrib['name']
            p = cx.findall('property')
            for e in p:
                xmlDat[cxn][e.attrib['name']] = e.findall('value')[0].text

        corGeoRscFile = corGeoFile + '.rsc'

        with open(corGeoRscFile,'w') as RSC:
            spc = " "*25
            for a in rscOrder:
                RSC.write(
                    "%s%s%s\n" % (a,spc[0:25-len(a)],xmlDat[rscXML[a][0]][rscXML[a][1]])
                    )
                
        return None

    def runGeocode(self, inFilename, widthIn, geoFilename, demcropFilename):
        import stdproc
        from isceobj import createDemImage

        print("runGeocode: inFilename, widthIn = ", inFilename, widthIn)
        print("runGeocode: geoFilename, demcropFilename = ", geoFilename, demcropFilename)

        logger.info("Geocoding Image")
                        
        # Initialize the Dem
        
        demImage = createDemImage()
        IU.copyAttributes(self.insar.demImage, demImage)
        demImage.setAccessMode('read')
        demImage.createImage()

        # Initialize the flattened interferogram
        from isceobj import createIntImage, createImage
        intImage = createIntImage()
        intImage.filename = inFilename
        intImage.width = widthIn
        intImage.setAccessMode('read')
        intImage.createImage()
        
        minLat, maxLat, minLon, maxLon = self.insar.topo.snwe

        planet = self.insar.referenceFrame.instrument.getPlatform().getPlanet()
             
        objGeo = stdproc.createGeocode() 
        objGeo.listInputPorts()
        objGeo.wireInputPort(name='peg',object=self.insar.peg)
        objGeo.wireInputPort(name='frame',object=self.insar.referenceFrame)
        objGeo.wireInputPort(name='planet',object=planet)
        objGeo.wireInputPort(name='dem',object=demImage)
        objGeo.wireInputPort(name='interferogram',object=intImage)
        objGeo.wireInputPort(name='geoPosting', object=self.geoPosting)
        print("self.geoPosting = ", self.geoPosting)

        objGeo.snwe = minLat, maxLat, minLon, maxLon
        objGeo.geoFilename = geoFilename
        objGeo.demCropFilename = demcropFilename

        #set the tag used in the outfile. each message is precided by this tag
        #is the writer is not of "file" type the call has no effect
        objGeo.stdWriter = self.stdWriter.set_file_tags("geocode", "log", "err", "out")

        # see mocompbaseline
        objFormSlc1  =  self.insar.formSLC1
        mocompPosition1 = objFormSlc1.getMocompPosition()
        posIndx = 1
        objGeo.referenceOrbit = mocompPosition1[posIndx]
        prf1 = self.insar.referenceFrame.instrument.getPulseRepetitionFrequency()
        dp = self.insar.dopplerCentroid.getDopplerCoefficients(inHz=False)[0]
        v = self.insar.procVelocity
        h = self.insar.averageHeight
        objGeo.setDopplerCentroidConstantTerm(dp)
        objGeo.setBodyFixedVelocity(v)
        objGeo.setSpacecraftHeight(h)
        objGeo.setNumberRangeLooks(self.insar.numberRangeLooks)
        objGeo.setNumberAzimuthLooks(self.insar.numberAzimuthLooks)
        # I have no idea what ismocomp means
        goodLines = self.insar.numberValidPulses 
        patchSize = self.insar.patchSize 
        # this variable was hardcoded in geocode.f90 and was equal to (8192 - 2048)/2
        is_mocomp = self.insar.is_mocomp
#        is_mocomp = int((patchSize - goodLines)/2)
        objGeo.setISMocomp(is_mocomp)

        objGeo.geocode()
       
        intImage.finalizeImage()
        demImage.finalizeImage()
        return objGeo

    def runGeocode4rlks(self, inFilename, widthIn, geoFilename, demcropFilename):
        import stdproc
        from isceobj import createIntImage, createImage        

        print("runGeocode4rlks: inFilename, widthIn = ", inFilename, widthIn)
        print("runGeocode4rlks: geoFilename, demcropFilename = ",
              geoFilename,
              demcropFilename)
        pause(message="Paused in runGeocode4rlks")
        
        logger.info("Geocoding Image")
                        
        # Initialize the Dem
        from isceobj import createDemImage
        demImage = createDemImage()
        IU.copyAttributes(self.insar.demImage,demImage)
        demImage.setAccessMode('read')
        demImage.createImage()
        print("demImage.firstLatitude = ", demImage.firstLatitude)
        print("demImage.firstLongitude = ", demImage.firstLongitude)
        print("demImage.deltaLatitude = ", demImage.deltaLatitude)
        print("demImage.deltaLongitude = ", demImage.deltaLongitude)
        print("demImage.width = ", demImage.width)
        print("demImage.length = ", demImage.length)
        demImage_lastLatitude = (
            demImage.firstLatitude + (demImage.length-1)*demImage.deltaLatitude
            )
        demImage_lastLongitude = (
            demImage.firstLongitude + (demImage.width-1)*demImage.deltaLongitude
            )

        print("demImage_lastLatitude = ", demImage_lastLatitude)
        print("demImage_lastLongitude = ", demImage_lastLongitude)
 
        # Initialize the input image
        intImage = createIntImage()
        intImage.setFilename(inFilename)
        intImage.setWidth(widthIn)
        intImage.setAccessMode('read')
        intImage.createImage()
               
        minLat, maxLat, minLon, maxLon = self.insar.topo.snwe
        print("objTopo.minLat = ", minLat)
        print("objTopo.minLon = ", minLon)
        print("objTopo.maxLat = ", maxLat)
        print("objTopo.maxLon = ", maxLon)
        pause(message="Paused in runGeocode4rlks")
        
        planet = self.insar.referenceFrame.instrument.getPlatform().getPlanet()
             
        objGeo = stdproc.createGeocode()       
        objGeo.listInputPorts()
        objGeo.wireInputPort(name='peg',object=self.insar.peg)
#        objGeo.wireInputPort(name='frame',object=self.insar.referenceFrame)
        objGeo.rangeFirstSample = self.insar.referenceFrame.getStartingRange()
        objGeo.slantRangePixelSpacing = self.insar.referenceFrame.instrument.getRangePixelSize()*4
        objGeo.prf = self.insar.referenceFrame.instrument.getPulseRepetitionFrequency()
        objGeo.radarWavelength = self.insar.referenceFrame.instrument.getRadarWavelength()
        objGeo.wireInputPort(name='planet',object=planet)
        objGeo.wireInputPort(name='dem',object=demImage)
        objGeo.wireInputPort(name='interferogram',object=intImage)
        print("self.geoPosting = ",self.geoPosting)
        objGeo.wireInputPort(name='geoPosting',object=self.geoPosting)

        objGeo.snwe = minLat, maxLat, minLon, maxLon
        objGeo.setGeocodeFilename(geoFilename)
        objGeo.setDemCropFilename(demcropFilename)

        #set the tag used in the outfile. each message is precided by this tag
        #is the writer is not of "file" type the call has no effect
        objGeo.stdWriter = self.stdWriter.set_file_tags("geocode", "log", "err", "out")

        # see mocompbaseline
        objFormSlc1  =  self.insar.formSLC1
        mocompPosition1 = objFormSlc1.getMocompPosition()
        posIndx = 1
        objGeo.setReferenceOrbit(mocompPosition1[posIndx])
        prf1 = self.insar.referenceFrame.instrument.getPulseRepetitionFrequency()
        dp = self.insar.dopplerCentroid.getDopplerCoefficients(inHz=False)[0]
        v = self.insar.procVelocity
        h = self.insar.averageHeight
        objGeo.setDopplerCentroidConstantTerm(dp)
        objGeo.setBodyFixedVelocity(v)
        objGeo.setSpacecraftHeight(h)
        objGeo.setNumberRangeLooks(1.0)  #self.insar.numberRangeLooks)
        objGeo.setNumberAzimuthLooks(1.0)  #self.insar.numberAzimuthLooks)
        # I have no idea what ismocomp means
        goodLines = self.insar.numberValidPulses 
        patchSize = self.insar.patchSize 
        # this variable was hardcoded in geocode.f90 and was equal to (8192 - 2048)/2
        is_mocomp = self.insar.is_mocomp
#        is_mocomp = int((patchSize - goodLines)/2)
        objGeo.setISMocomp(is_mocomp)

        objGeo.geocode()

        print("Input state paraemters to gecode.f90:")
        print("Minimum Latitude = ", objGeo.minimumLatitude)
        print("Maximum Latitude = ", objGeo.maximumLatitude)
        print("Minimum Longitude = ", objGeo.minimumLongitude)
        print("Maximum Longitude = ", objGeo.maximumLongitude)
        print("Ellipsoid Major Semi Axis = ", objGeo.ellipsoidMajorSemiAxis)
        print("Ellipsoid Eccentricity Squared = ", objGeo.ellipsoidEccentricitySquared)
        print("Peg Latitude = ", objGeo.pegLatitude)
        print("Peg Longitude = ", objGeo.pegLongitude)
        print("Peg Heading = ", objGeo.pegHeading)
        print("Range Pixel Spacing = ", objGeo.slantRangePixelSpacing)
        print("Range First Sample  = ", objGeo.rangeFirstSample)
        print("Spacecraft Height  = ", objGeo.spacecraftHeight)
        print("Planet Local Radius  = ", objGeo.planetLocalRadius)
        print("Body Fixed Velocity  = ", objGeo.bodyFixedVelocity)
        print("Doppler Centroid Constant Term  = ", objGeo.dopplerCentroidConstantTerm)
        print("PRF  = ", objGeo.prf)
        print("Radar Wavelength  = ", objGeo.radarWavelength)
        print("S Coordinate First Line  = ", objGeo.sCoordinateFirstLine)
        print("Azimuth Spacing = ", objGeo.azimuthSpacing)
        print("First Latitude  = ", objGeo.firstLatitude)
        print("First Longitude  = ", objGeo.firstLongitude)
        print("Delta Latitude  = ", objGeo.deltaLatitude)
        print("Delta Longitude  = ", objGeo.deltaLongitude)
        print("Length  = ", objGeo.length)
        print("Width  = ", objGeo.width)
        print("Number Range Looks  = ", objGeo.numberRangeLooks)
        print("Number Azimuth Looks  = ", objGeo.numberAzimuthLooks)
        print("Number Points Per DEM Post  = ", objGeo.numberPointsPerDemPost)
        print("Is Mocomp  = ", objGeo.isMocomp)
        print("DEM Width  = ", objGeo.demWidth)
        print("DEM Length  = ", objGeo.demLength)
#        print("Reference Orbit  = ", objGeo.referenceOrbit)
        print("Dim1 Reference Orbit  = ", objGeo.dim1_referenceOrbit)
        intImage.finalizeImage()
        demImage.finalizeImage()
        return objGeo


    def runGeocodeCor(self):
        import stdproc
        
        logger.info("Geocoding Correlation")
        objFormSlc1  =  self.insar.formSLC1
        # Initialize the Dem
        from isceobj import createDemImage, createIntImage, createImage
        demImage = createDemImage()
        IU.copyAttributes(self.insar.demImage,demImage)
        demImage.setAccessMode('read')
        demImage.createImage()

        topoflatIntFilename = self.insar.topophaseFlatFilename
        widthInt = self.insar.resampIntImage.width
        
        intImage = createIntImage()
        widthInt = self.insar.resampIntImage.width
        intImage.setFilename(corintFilename)
        intImage.setWidth(widthInt)
        intImage.setAccessMode('read')
        intImage.createImage()
        
        posIndx = 1
        mocompPosition1 = objFormSlc1.getMocompPosition()
                   
        minLat, maxLat, minLon, maxLon = self.insar.topo.snwe

        planet = self.insar.referenceFrame.instrument.getPlatform().getPlanet()
             
        objGeo = stdproc.createGeocode()       
        objGeo.wireInputPort(name='peg',object=self.insar.peg)
        objGeo.wireInputPort(name='frame',object=self.insar.referenceFrame)
        objGeo.wireInputPort(name='planet',object=planet)
        objGeo.wireInputPort(name='dem',object=demImage)
        objGeo.wireInputPort(name='interferogram',object=intImage)
        objGeo.snwe = minLat, maxLat, minLon, maxLon
        corGeocodeFilename = corintFilename+'.geo'
        demGeocodeFilename = corintFilename+'.demcrop'
        objGeo.setGeocodeFilename(corGeocodeFilename)
        objGeo.setDemCropFilename(demGeocodeFilename)
        #set the tag used in the outfile. each message is precided by this tag
        #is the writer is not of "file" type the call has no effect
        objGeo.stdWriter = self.stdWriter.set_file_tags("geocode", "log", "err", "out")
        # see mocompbaseline
        objGeo.setReferenceOrbit(mocompPosition1[posIndx])
        prf1 = self.insar.referenceFrame.instrument.getPulseRepetitionFrequency()
        dp = self.insar.dopplerCentroid.getDopplerCoefficients(inHz=False)[0]
        v = self.insar.procVelocity
        h = self.insar.averageHeight
        objGeo.setDopplerCentroidConstantTerm(dp)
        objGeo.setBodyFixedVelocity(v)
        objGeo.setSpacecraftHeight(h)
        objGeo.setNumberRangeLooks(self.insar.numberRangeLooks)
        objGeo.setNumberAzimuthLooks(self.insar.numberAzimuthLooks)
        # I have no idea what ismocomp means
        goodLines = self.insar.numberValidPulses 
        patchSize = self.insar.patchSize 
        # this variable was hardcoded in geocode.f90 and was equal to (8192 - 2048)/2
        is_mocomp = int((patchSize - goodLines)/2)
        objGeo.setISMocomp(is_mocomp)

        objGeo.geocode()
       
        intImage.finalizeImage()
        demImage.finalizeImage()
        return objGeo


    def restart(self):
        print("Restarting with Filtering")
        return
    
    ## main() extends _InsarBase.main()
    def main(self):
        import time
        timeStart = time.time()
        
        super(Dpm, self).main()

#        self.runCorrect()
        
        self.runShadecpx2rg()
        
        self.runRgoffset()

        # Cull offoutliers
        self.iterate_runOffoutliers()

        self.runResamp_only()

        self.insar.topoIntImage=self.insar.resampOnlyImage
        self.runTopo()
        self.runCorrect()
        
        # Coherence ?
        self.runCoherence(method=self.correlation_method)

        #ouput the procDoc and pause in order to process coherence off line
        #this processing should really be done using _steps.
        self.insar.procDoc.renderXml()
        pause(message="Paused in main")

        # Filter ?
        self.runFilter()
    
        # Unwrap ?
        self.verifyUnwrap()

        # Geocode
        self.geocodeCorIfg()

        timeEnd = time.time()
        logger.info("Total Time: %i seconds" %(timeEnd - timeStart))

        self.insar.procDoc.renderXml()
        
        return None


def rmgcor_ifgphs_to_cmplx(rmg,ifg,cpx,width):
    import struct
    import math

    raise DeprecationWarning("Don't ude this function")

    length = int(os.stat(ifg).st_size/8./width)

    rmgFile = open(rmg,'rb')
    ifgFile = open(ifg,'rb')
    cpxFile = open(cpx,'wb')

    w = int(width)
    width2 = 2*w
    fmt = "%df" % (width2,)
    aCpxLine = [0.0]*width2

    for iii in range(length):
        anIfgLine = struct.unpack(fmt,ifgFile.read(width2*4))
        aRmgLine  = struct.unpack(fmt,rmgFile.read(width2*4))
        for jjj in range(w):
            ifgPhase = math.atan2(anIfgLine[2*jjj+1],anIfgLine[2*jjj])
#            ampVal = aRmgLine[jjj]
            corVal = aRmgLine[w+jjj]
            aCpxLine[2*jjj] = corVal*math.cos(ifgPhase)
            aCpxLine[2*jjj+1] = corVal*math.sin(ifgPhase)
        cpxFile.write(struct.pack(fmt,*aCpxLine))

    rmgFile.close()
    ifgFile.close()
    cpxFile.close()
    return

def ifg1amp_ifg2amp_to_rmg(ifg1,ifg2,rmg,width):
    import struct
    import math

    raise DeprecationWarning("Don't ude this function")

    length = int(os.stat(ifg1).st_size/8./width)

    ifg1File = open(ifg1,'rb')
    ifg2File = open(ifg2,'rb')
    rmgFile = open(rmg,'wb')

    w = int(width)
    width2 = 2*w
    fmt = "%df" % (width2,)
    aRmgLine = [0.0]*width2

    for iii in range(length):
        anIfg1Line = struct.unpack(fmt,ifg1File.read(width2*4))
        anIfg2Line = struct.unpack(fmt,ifg2File.read(width2*4))
        for jjj in range(w):
            amp1 = math.sqrt(anIfg1Line[2*jjj]**2 + anIfg1Line[2*jjj+1]**2)
            amp2 = math.sqrt(anIfg2Line[2*jjj]**2 + anIfg2Line[2*jjj+1]**2)
            aRmgLine[jjj] = amp1
            aRmgLine[w + jjj] = amp2
        rmgFile.write(struct.pack(fmt,*aRmgLine))

    ifg1File.close()
    ifg2File.close()
    rmgFile.close()
    return

def rmg_to_cmplx(rmg,cpx,width):
    import struct
    import math 

    length = int(os.stat(rmg).st_size/8./width)

    rmgFile = open(rmg,'rb')
    cpxFile = open(cpx,'wb')

    w = int(width)
    width2 = 2*w
    fmt = "%df" % (width2,)
    aCpxLine = [0.0]*width2

    for iii in range(length):
        aRmgLine  = struct.unpack(fmt,rmgFile.read(width2*4))
        for jjj in range(w):
            ampVal = aRmgLine[jjj]
            corVal = aRmgLine[w+jjj]
            aCpxLine[2*jjj] = ampVal
            aCpxLine[2*jjj+1] = corVal
        cpxFile.write(struct.pack(fmt,*aCpxLine))

    rmgFile.close()
    cpxFile.close()
    return

def cmplx_to_rmg(ifg1,rmg,width):
    import struct
    import math 

    length = int(os.stat(ifg1).st_size/8./width)

    ifg1File = open(ifg1,'rb')
    rmgFile = open(rmg,'wb')

    w = int(width)
    width2 = 2*w
    fmt = "%df" % (width2,)
    aRmgLine = [0.0]*width2

    for iii in range(length):
        anIfg1Line = struct.unpack(fmt,ifg1File.read(width2*4))
        for jjj in range(w):
            amp1 = anIfg1Line[2*jjj]
            amp2 = anIfg1Line[2*jjj+1]
            aRmgLine[jjj] = amp1 
            aRmgLine[w + jjj] = amp2
        rmgFile.write(struct.pack(fmt,*aRmgLine))

    ifg1File.close()
    rmgFile.close()
    return

    

if __name__ == "__main__":
    dpm = Dpm()
    dpm.run()
    
