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




try:
    import h5py
except ImportError:
    raise Exception("Python module h5py is required to process Generic data")
import datetime
import logging
import isceobj
import numpy
from isceobj.Scene.Frame import Frame
from isceobj.Scene.Track import Track
from isceobj.Orbit.Orbit import Orbit, StateVector
from isceobj.Planet.Planet import Planet
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTU
from iscesys.Component.Component import Component

class Generic(Component):
    """
    A class to parse generic SAR data stored in the HDF5 format
    """

    logging_name =  'isce.sensor.Generic'

    def __init__(self):
        super(Generic, self).__init__()
        self._hdf5File = None
        self.output = None
        self.frame = Frame()
        self.frame.configure()

        self.logger = logging.getLogger('isce.sensor.Generic')

        self.descriptionOfVariables = {}
        self.dictionaryOfVariables = {
            'HDF5': ['self._hdf5File','str','mandatory'],
            'OUTPUT': ['self.output','str','optional']}
        return None

    
    def getFrame(self):
        return self.frame

    def parse(self):
        try:
            fp = h5py.File(self._hdf5File,'r')
        except Exception as strerror:
            self.logger.error("IOError: %s" % strerror)
            return

        self.populateMetadata(fp)
        fp.close()

    def populateMetadata(self,file):
        """
        Create the appropriate metadata objects from our HDF5 file
        """
        self._populatePlatform(file)
        self._populateInstrument(file)
        self._populateFrame(file)
        self._populateOrbit(file)

    def _populatePlatform(self,file):
        platform = self.frame.getInstrument().getPlatform()

        platform.setMission(file['Platform'].attrs['Mission'])
        platform.setPlanet(Planet(pname='Earth'))
        platform.setAntennaLength(file['Platform'].attrs['Antenna Length'])

    def _populateInstrument(self,file):
        instrument = self.frame.getInstrument()

        instrument.setRadarWavelength(file['Instrument'].attrs['Wavelength'])
        instrument.setPulseRepetitionFrequency(file['Instrument'].attrs['Pulse Repetition Frequency'])
        instrument.setRangePixelSize(file['Instrument'].attrs['Range Pixel Size'])
        instrument.setPulseLength(file['Instrument'].attrs['Pulse Length'])
        instrument.setChirpSlope(file['Instrument'].attrs['Chirp Slope'])
        instrument.setRangeSamplingRate(file['Instrument'].attrs['Range Sampling Frequency'])
        instrument.setInPhaseValue(file['Frame'].attrs['In Phase Bias'])
        instrument.setQuadratureValue(file['Frame'].attrs['Quadrature Bias'])

    def _populateFrame(self,file):
        size = file['Frame'].shape
        start = DTU.parseIsoDateTime(file['Frame'].attrs['Sensing Start'])
        stop = DTU.parseIsoDateTime(file['Frame'].attrs['Sensing Stop'])
        deltaT = DTU.timeDeltaToSeconds(stop-start)
        mid = start + datetime.timedelta(microseconds=int(deltaT/2.0*1e6))
        startingRange = file['Frame'].attrs['Starting Range']
        rangePixelSize = file['Instrument'].attrs['Range Pixel Size']
        farRange = startingRange + size[1]*rangePixelSize

        self.frame.setStartingRange(file['Frame'].attrs['Starting Range'])
        self.frame.setFarRange(farRange)
        self.frame.setNumberOfLines(size[0])
        self.frame.setNumberOfSamples(2*size[1])
        self.frame.setSensingStart(start)
        self.frame.setSensingMid(mid)
        self.frame.setSensingStop(stop)

    def _populateOrbit(self,file):
        orbit = self.frame.getOrbit()

        orbit.setReferenceFrame('ECR')
        orbit.setOrbitSource(file['Orbit'].attrs['Source'])

        for i in range(len(file['Orbit']['Time'])):
            vec = StateVector()
            time = DTU.parseIsoDateTime(file['Orbit']['Time'][i])
            vec.setTime(time)
            vec.setPosition(list(file['Orbit']['Position'][i]))
            vec.setVelocity(list(file['Orbit']['Velocity'][i]))
            orbit.addStateVector(vec)

    def extractImage(self):
        try:
            file = h5py.File(self._hdf5File,'r')
        except Exception as strerror:
            self.logger.error("IOError: %s" % strerror)
            return
        size = file['Frame'].shape
        dtype = self._translateDataType(file['Frame'].attrs['Image Type'])
        length = size[0]
        width = size[1]
        data = numpy.memmap(self.output, dtype=dtype, mode='w+', shape=(length,width,2))
        data[:,:,:] =  file['Frame'][:,:,:]
        del data

        rawImage = isceobj.createRawImage()
        rawImage.setByteOrder('l')
        rawImage.setAccessMode('r')
        rawImage.setFilename(self.output)
        rawImage.setWidth(2*width)
        rawImage.setXmin(0)
        rawImage.setXmax(2*width)
        self.frame.setImage(rawImage)
        self.populateMetadata(file)

        file.close()

    def write(self,output,compression=None):
        """
        Given a frame object (appropriately populated) and an image, create
        an HDF5 from those objects.
        """
        if (not self.frame):
            self.logger.error("Frame not set")
            raise AttributeError("Frame not set")

        h5file = h5py.File(output,'w')
        self._writeMetadata(h5file,compression)

    def _writeMetadata(self,h5file,compression=None):
        self._writePlatform(h5file)
        self._writeInstrument(h5file)
        self._writeFrame(h5file,compression)
        self._writeOrbit(h5file)

    def _writePlatform(self,h5file):
        platform = self.frame.getInstrument().getPlatform()
        if (not platform):
            self.logger.error("Platform not set")
            raise AttributeError("Platform not set")

        group = h5file.create_group('Platform')
        group.attrs['Mission'] = platform.getMission()
        group.attrs['Planet'] = platform.getPlanet().name
        group.attrs['Antenna Length'] = platform.getAntennaLength()

    def _writeInstrument(self,h5file):
        instrument = self.frame.getInstrument()
        if (not instrument):
            self.logger.error("Instrument not set")
            raise AttributeError("Instrument not set")

        group = h5file.create_group('Instrument')
        group.attrs['Wavelength'] = instrument.getRadarWavelength()
        group.attrs['Pulse Repetition Frequency'] = instrument.getPulseRepetitionFrequency()
        group.attrs['Range Pixel Size'] = instrument.getRangePixelSize()
        group.attrs['Pulse Length'] = instrument.getPulseLength()
        group.attrs['Chirp Slope'] = instrument.getChirpSlope()
        group.attrs['Range Sampling Frequency'] = instrument.getRangeSamplingRate()
        group.attrs['In Phase Bias'] = instrument.getInPhaseValue()
        group.attrs['Quadrature Bias'] = instrument.getQuadratureValue()

    def _writeFrame(self,h5file,compression=None):
        group = self._writeImage(h5file,compression)
        group.attrs['Starting Range'] = self.frame.getStartingRange()
        group.attrs['Sensing Start'] = self.frame.getSensingStart().isoformat()
        group.attrs['Sensing Stop'] = self.frame.getSensingStop().isoformat()

    def _writeImage(self,h5file,compression=None):
        image = self.frame.getImage()
        if (not image):
            self.logger.error("Image not set")
            raise AttributeError("Image not set")
        filename = image.getFilename()
        length = image.getLength()
        width = image.getWidth()

        dtype = self._translateDataType(image.dataType)
        if (image.dataType == 'BYTE'):
            width = int(width/2)

        self.logger.debug("Width: %s" % (width))
        self.logger.debug("Length: %s" % (length))
        data = numpy.memmap(filename, dtype=dtype, mode='r', shape=(length,2*width))
        dset = h5file.create_dataset("Frame",(length,width,2),dtype=dtype,compression=compression)
        dset.attrs['Image Type'] = image.dataType
        dset[:,:,0] = data[:,::2]
        dset[:,:,1] = data[:,1::2]
        del data

        return dset

    def _writeOrbit(self,h5file):
        # Add orbit information
        (time,position,velocity) = self._orbitToArray()
        group = h5file.create_group("Orbit")
        group.attrs['Source'] = self.frame.getOrbit().getOrbitSource()
        group.attrs['Reference Frame'] = self.frame.getOrbit().getReferenceFrame()
        timedset = h5file.create_dataset("Orbit/Time",time.shape,dtype=time.dtype)
        posdset = h5file.create_dataset("Orbit/Position", position.shape,dtype=numpy.float64)
        veldset = h5file.create_dataset("Orbit/Velocity", velocity.shape,dtype=numpy.float64)
        timedset[...] = time
        posdset[...] = position 
        veldset[...] = velocity

    def _orbitToArray(self):
        orbit = self.frame.getOrbit()
        if (not orbit):
            self.logger.error("Orbit not set")
            raise AttributeError("Orbit not set")

        time  = []
        position = []
        velocity = []
        for sv in orbit:
            timeString = sv.getTime().isoformat()
            time.append(timeString)
            position.append(sv.getPosition())
            velocity.append(sv.getVelocity())

        return numpy.array(time), numpy.array(position), numpy.array(velocity)

    def _translateDataType(self,imageType):
        dtype = ''
        if (imageType == 'BYTE'):
            dtype = 'int8'
        elif (imageType == 'CFLOAT'):
            dtype = 'float32'
        elif (imageType == 'SHORT'):
            dtype = 'int16'
        else:
            self.logger.error("Unknown data type %s" % (imageType))
            raise ValueError("Unknown data type %s" % (imageType))

        return dtype
