#!/usr/bin/env python3

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2025 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Scott Staniewicz
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import datetime
import json
import os
import numpy as np

import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector, Orbit
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil

lookMap = {'RIGHT': -1, 'LEFT': 1}

TIFF = Component.Parameter(
    'tiff',
    public_name='TIFF',
    default='',
    type=str,
    mandatory=True,
    doc='Capella SLC GeoTIFF imagery file'
)

METADATA = Component.Parameter(
    'metadataFile',
    public_name='METADATA',
    default='',
    type=str,
    mandatory=True,
    doc='Capella extended JSON metadata file'
)

from .Sensor import Sensor


class Capella(Sensor):
    """
    A class representing Capella Space SAR SLC data.
    """

    family = 'capella'
    logging_name = 'isce.sensor.Capella'
    parameter_list = (TIFF, METADATA) + Sensor.parameter_list

    def __init__(self, family='', name=''):
        super().__init__(family if family else self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._metadata = None
        self.doppler_coeff = None

    def getFrame(self):
        return self.frame

    def parse(self):
        """Parse the Capella metadata JSON file."""
        try:
            with open(self.metadataFile, 'r') as fp:
                self._metadata = json.load(fp)
        except IOError as strerr:
            print("IOError: %s" % strerr)
            return
        self.populateMetadata()

    def populateMetadata(self):
        """Create metadata objects from the JSON metadata."""

        collect = self._metadata.get('collect', {})
        radar = self._metadata.get('radar', {})
        state_vectors = self._metadata.get('state_vectors', [])
        image = collect.get('image', {})

        # Platform info
        mission = collect.get('platform', 'Capella')
        pointing = radar.get('pointing', 'right')
        lookSide = lookMap.get(pointing.upper(), -1)

        # Radar parameters
        frequency = radar.get('center_frequency', 9.65e9)
        txPol = radar.get('transmit_polarization', 'H')
        rxPol = radar.get('receive_polarization', 'H')
        polarization = txPol + rxPol

        # Get PRF from time_varying_parameters or prf array
        prf_list = radar.get('prf', [])
        if prf_list:
            prf = prf_list[0].get('prf', 3000.0)
        else:
            tvp = radar.get('time_varying_parameters', [])
            if tvp:
                prf = tvp[0].get('prf', 3000.0)
            else:
                prf = 3000.0

        # Pulse parameters from time_varying_parameters
        tvp = radar.get('time_varying_parameters', [])
        if tvp:
            pulseBandwidth = tvp[0].get('pulse_bandwidth', 500e6)
            pulseLength = tvp[0].get('pulse_duration', 10e-6)
        else:
            pulseBandwidth = 500e6
            pulseLength = 10e-6

        # Sampling rate
        samplingFrequency = radar.get('sampling_frequency', 600e6)

        # Image dimensions
        lines = image.get('rows', image.get('length', 0))
        samples = image.get('columns', image.get('width', 0))

        # Pixel spacing
        rangePixelSize = image.get('pixel_spacing_column', 0.5)
        azimuthPixelSize = image.get('pixel_spacing_row', 0.5)

        # Image geometry for starting range
        image_geometry = image.get('image_geometry', {})
        range_time_origin = image_geometry.get('range_time_origin', 0.0)
        startingRange = range_time_origin * Const.c / 2.0

        # Incidence angle
        center_pixel = image.get('center_pixel', {})
        incidenceAngle = center_pixel.get('incidence_angle', 35.0)

        # Timing information
        startTimeStr = collect.get('start_timestamp', '')
        stopTimeStr = collect.get('stop_timestamp', '')
        dataStartTime = self._parseDateTime(startTimeStr)
        dataStopTime = self._parseDateTime(stopTimeStr)

        # Pass direction from state vectors
        if len(state_vectors) >= 2:
            z0 = state_vectors[0].get('position', [0, 0, 0])[2]
            z1 = state_vectors[-1].get('position', [0, 0, 0])[2]
            passDirection = 'Ascending' if z1 > z0 else 'Descending'
        else:
            passDirection = 'Descending'

        # Populate platform
        platform = self.frame.getInstrument().getPlatform()
        platform.setPlanet(Planet(pname="Earth"))
        platform.setMission(mission)
        platform.setPointingDirection(lookSide)
        platform.setAntennaLength(3.0)  # Capella uses ~3m antenna

        # Populate instrument
        instrument = self.frame.getInstrument()
        instrument.setRadarFrequency(frequency)
        instrument.setPulseRepetitionFrequency(prf)
        instrument.setPulseLength(pulseLength)
        instrument.setChirpSlope(pulseBandwidth / pulseLength)
        instrument.setIncidenceAngle(incidenceAngle)
        instrument.setRangePixelSize(rangePixelSize)
        instrument.setRangeSamplingRate(samplingFrequency)

        # Populate Frame
        self.frame.setSensingStart(dataStartTime)
        self.frame.setSensingStop(dataStopTime)
        diffTime = DTUtil.timeDeltaToSeconds(dataStopTime - dataStartTime) / 2.0
        sensingMid = dataStartTime + datetime.timedelta(microseconds=int(diffTime * 1e6))
        self.frame.setSensingMid(sensingMid)
        self.frame.setPassDirection(passDirection)
        self.frame.setPolarization(polarization)
        self.frame.setStartingRange(startingRange)
        self.frame.setFarRange(startingRange + (samples - 1) * rangePixelSize)
        self.frame.setNumberOfLines(lines)
        self.frame.setNumberOfSamples(samples)
        self.frame.setProcessingFacility('Capella Space')
        self.frame.setProcessingSoftwareVersion(self._metadata.get('software_version', ''))

        # Extract orbit
        self.extractOrbit(state_vectors)

        # Save Doppler centroid coefficients
        # Capella provides doppler_centroid_polynomial in the image_geometry
        dc_poly = image_geometry.get('doppler_centroid_polynomial', [0.0])
        if isinstance(dc_poly, list) and len(dc_poly) > 0:
            self.doppler_coeff = dc_poly
        else:
            self.doppler_coeff = [0.0]

    def _parseDateTime(self, timeStr):
        """Parse Capella timestamp string to datetime object."""
        if not timeStr:
            return datetime.datetime.now()

        # Handle various timestamp formats
        # e.g., "2025-10-31T19:11:04.123456Z" or "2025-10-31T19:11:04Z"
        try:
            # Try with microseconds
            if '.' in timeStr:
                # Remove 'Z' suffix if present
                timeStr = timeStr.rstrip('Z')
                # Truncate to 6 decimal places for microseconds
                parts = timeStr.split('.')
                if len(parts) == 2:
                    timeStr = parts[0] + '.' + parts[1][:6]
                return datetime.datetime.strptime(timeStr, "%Y-%m-%dT%H:%M:%S.%f")
            else:
                timeStr = timeStr.rstrip('Z')
                return datetime.datetime.strptime(timeStr, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            print(f"Warning: Could not parse timestamp {timeStr}")
            return datetime.datetime.now()

    def extractOrbit(self, state_vectors):
        """Extract orbit state vectors from the metadata."""
        orbit = self.frame.getOrbit()
        orbit.setOrbitSource('Header')
        orbit.setReferenceFrame('ECR')

        for sv in state_vectors:
            vec = StateVector()
            timeStr = sv.get('time', '')
            vec.setTime(self._parseDateTime(timeStr))
            position = sv.get('position', [0, 0, 0])
            velocity = sv.get('velocity', [0, 0, 0])
            vec.setPosition(position)
            vec.setVelocity(velocity)
            orbit.addStateVector(vec)

    def extractImage(self):
        """
        Use GDAL to extract the SLC from Capella GeoTIFF.
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for Capella data.')

        self.parse()

        width = self.frame.getNumberOfSamples()
        lgth = self.frame.getNumberOfLines()

        src = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)

        # Capella SLC GeoTIFFs are complex int16 (CInt16) stored as a single band
        # or as two bands (real/imag)
        nbands = src.RasterCount

        if nbands == 1:
            # Single band complex
            data = src.GetRasterBand(1).ReadAsArray(0, 0, width, lgth)
            if data.dtype != np.complex64:
                data = data.astype(np.complex64)
        elif nbands == 2:
            # Two bands: real and imaginary
            real = src.GetRasterBand(1).ReadAsArray(0, 0, width, lgth)
            imag = src.GetRasterBand(2).ReadAsArray(0, 0, width, lgth)
            cJ = np.complex64(1.0j)
            data = real.astype(np.float32) + cJ * imag.astype(np.float32)
        else:
            raise Exception(f'Unexpected number of bands in Capella SLC: {nbands}')

        src = None

        data.tofile(self.output)

        # Create SLC image object
        slcImage = isceobj.createSlcImage()
        slcImage.setByteOrder('l')
        slcImage.setFilename(self.output)
        slcImage.setAccessMode('read')
        slcImage.setWidth(width)
        slcImage.setLength(lgth)
        slcImage.setXmin(0)
        slcImage.setXmax(width)
        self.frame.setImage(slcImage)

    def extractDoppler(self):
        """
        Extract Doppler centroid information.
        """
        quadratic = {}

        # For insarApp style
        prf = self.frame.getInstrument().getPulseRepetitionFrequency()
        if self.doppler_coeff and len(self.doppler_coeff) > 0:
            fd_mid = self.doppler_coeff[0]
        else:
            fd_mid = 0.0

        quadratic['a'] = fd_mid / prf
        quadratic['b'] = 0.
        quadratic['c'] = 0.

        # For roiApp - doppler vs pixel
        self.frame._dopplerVsPixel = self.doppler_coeff if self.doppler_coeff else [0.0]
        print('Doppler Fit: ', self.frame._dopplerVsPixel)

        return quadratic
