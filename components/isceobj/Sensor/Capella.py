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
import numpy as np

import isceobj
from isceobj.Scene.Frame import Frame
from isceobj.Planet.Planet import Planet
from isceobj.Orbit.Orbit import StateVector
from isceobj.Planet.AstronomicalHandbook import Const
from iscesys.Component.Component import Component
from iscesys.DateTimeUtil.DateTimeUtil import DateTimeUtil as DTUtil

from .Sensor import Sensor

lookMap = {'RIGHT': -1, 'LEFT': 1}

# Supported modes for stripmapStack processing
SUPPORTED_MODES = {'stripmap', 'SM', 'spotlight', 'SP'}

TIFF = Component.Parameter(
    'tiff',
    public_name='TIFF',
    default='',
    type=str,
    mandatory=True,
    doc='Capella SLC GeoTIFF imagery file'
)


class Capella(Sensor):
    """
    A class representing Capella Space SAR SLC data.

    Metadata is read from the TIFF ImageDescription tag (embedded JSON).
    Only stripmap (SM) mode is currently supported for stripmapStack processing.
    """

    family = 'capella'
    logging_name = 'isce.sensor.Capella'
    parameter_list = (TIFF,) + Sensor.parameter_list

    def __init__(self, family='', name=''):
        super().__init__(family if family else self.__class__.family, name=name)
        self.frame = Frame()
        self.frame.configure()
        self._metadata = None
        self.doppler_coeff = None

    def getFrame(self):
        return self.frame

    def _loadMetadataFromTiff(self):
        """Load metadata from TIFF ImageDescription tag using GDAL.

        Uses the actual TIFF dimensions (from GDAL) for image size rather than
        the metadata rows/columns, since properly cropped TIFFs (e.g. from
        sarlet crop-slc) update the timing/range metadata but may not update
        the rows/columns fields.
        """
        try:
            from osgeo import gdal
        except ImportError:
            raise Exception('GDAL python bindings not found. Need this for Capella data.')

        ds = gdal.Open(self.tiff.strip(), gdal.GA_ReadOnly)
        if ds is None:
            raise Exception(f'Could not open TIFF file: {self.tiff}')

        # Get ImageDescription from TIFF metadata
        image_desc = ds.GetMetadataItem('TIFFTAG_IMAGEDESCRIPTION')
        self._tiff_width = ds.RasterXSize
        self._tiff_height = ds.RasterYSize
        ds = None

        if not image_desc:
            raise Exception(f'No ImageDescription tag found in TIFF: {self.tiff}')

        try:
            self._metadata = json.loads(image_desc)
        except json.JSONDecodeError as e:
            raise Exception(f'Failed to parse JSON from TIFF ImageDescription: {e}')

    def parse(self):
        """Parse the Capella metadata from TIFF ImageDescription tag."""
        self._loadMetadataFromTiff()
        self._validateMode()
        self.populateMetadata()

    def _validateMode(self):
        """Validate that the acquisition mode is supported."""
        collect = self._metadata.get('collect', {})
        mode = collect.get('mode', '')

        if mode.lower() not in {m.lower() for m in SUPPORTED_MODES}:
            raise Exception(
                f"Capella mode '{mode}' is not supported for stripmapStack. "
                f"Supported modes: {SUPPORTED_MODES}. "
                f"Sliding Spotlight (SL) mode is not yet supported."
            )

    def populateMetadata(self):
        """Create metadata objects from the JSON metadata."""

        collect = self._metadata.get('collect', {})
        # Radar and state vectors can be at top level (extended JSON) or inside collect (TIFF embedded)
        radar = self._metadata.get('radar', collect.get('radar', {}))
        state_obj = self._metadata.get('state_vectors', collect.get('state', {}).get('state_vectors', []))
        # Handle both list format and nested state object
        state_vectors = state_obj if isinstance(state_obj, list) else []
        image = collect.get('image', {})

        # Platform info
        mission = collect.get('platform', 'Capella')
        pointing = radar.get('pointing', 'right')
        lookSide = lookMap.get(pointing.upper(), -1)

        # Radar parameters
        frequency = radar.get('center_frequency', 9.65e9)
        txPol = radar.get('transmit_polarization', 'V')
        rxPol = radar.get('receive_polarization', 'V')
        polarization = txPol + rxPol

        # Pulse parameters from time_varying_parameters
        tvp = radar.get('time_varying_parameters', [])
        if tvp:
            pulseBandwidth = tvp[0].get('pulse_bandwidth', 500e6)
            pulseLength = tvp[0].get('pulse_duration', 10e-6)
        else:
            pulseBandwidth = 500e6
            pulseLength = 10e-6

        # Image dimensions — use actual TIFF size (handles cropped TIFFs correctly)
        lines = self._tiff_height
        samples = self._tiff_width

        # Image geometry defines the actual SLC pixel grid
        # These are the ground truth for timing and range spacing
        image_geometry = image.get('image_geometry', {})
        startingRange = image_geometry.get('range_to_first_sample', 0.0)
        delta_range_sample = image_geometry.get('delta_range_sample', 0.0)
        delta_line_time = image_geometry.get('delta_line_time', 0.0)
        assert delta_range_sample > 0, f"Missing delta_range_sample in image_geometry"
        assert delta_line_time > 0, f"Missing delta_line_time in image_geometry"

        # PRF = 1/delta_line_time (the processed SLC line rate, not the raw radar PRF)
        prf = 1.0 / delta_line_time

        # Slant range pixel size from the SLC grid (not ground pixel spacing)
        rangePixelSize = delta_range_sample

        # Sampling rate consistent with SLC range pixel spacing
        samplingFrequency = Const.c / (2.0 * delta_range_sample)

        # Incidence angle
        center_pixel = image.get('center_pixel', {})
        incidenceAngle = center_pixel.get('incidence_angle', 35.0)

        # Timing: use first_line_time from image_geometry (actual SLC grid start),
        # NOT collect start/stop timestamps (which span the full acquisition window)
        firstLineTimeStr = image_geometry.get('first_line_time', '')
        assert firstLineTimeStr, "Missing first_line_time in image_geometry"
        dataStartTime = self._parseDateTime(firstLineTimeStr)
        dataStopTime = dataStartTime + datetime.timedelta(
            seconds=(lines - 1) * delta_line_time
        )

        # Pass direction from state vectors
        if len(state_vectors) >= 2:
            pos0 = state_vectors[0].get('position', [0, 0, 0])
            pos1 = state_vectors[-1].get('position', [0, 0, 0])
            z0 = pos0['z'] if isinstance(pos0, dict) else pos0[2]
            z1 = pos1['z'] if isinstance(pos1, dict) else pos1[2]
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
        sensingMid = dataStartTime + datetime.timedelta(
            seconds=(lines - 1) * delta_line_time / 2.0
        )
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

        # Extract Doppler centroid coefficients
        # Capella provides a 2D polynomial (azimuth, range), but isce2 needs 1D (range only)
        # We evaluate the 2D poly at mid-azimuth to get 1D coefficients vs range pixel
        self.doppler_coeff = self._extractDopplerCoeffs(image_geometry, lines, samples)

        # Spotlight-specific metadata for phase correction
        # reference_antenna_position and reference_target_position are ECEF coordinates
        # used to compute the geometric phase that must be restored after coregistration
        mode = collect.get('mode', '')
        if mode.lower() in {'spotlight', 'sp'}:
            ref_ant = image.get('reference_antenna_position')
            ref_tgt = image.get('reference_target_position')
            assert ref_ant, "Spotlight mode requires reference_antenna_position in metadata"
            assert ref_tgt, "Spotlight mode requires reference_target_position in metadata"
            # Positions are [x, y, z] ECEF coordinates in meters
            self.frame.spotlightReferenceAntennaPosition = list(ref_ant)
            self.frame.spotlightReferenceTargetPosition = list(ref_tgt)

    def _extractDopplerCoeffs(self, image_geometry, lines, samples):
        """
        Extract 1D Doppler coefficients from Capella's 2D polynomial.

        Capella provides a 2D polynomial: doppler(az, rg) = sum_{i,j} c[i][j] * az^i * rg^j
        where az and rg are in pixel coordinates.

        ISCE2 expects a 1D polynomial as a function of range pixel.
        We evaluate at mid-azimuth to get 1D coeffs.
        """
        dc_poly = image_geometry.get('doppler_centroid_polynomial', {})

        # Handle case where it's not a dict (legacy or missing)
        if not isinstance(dc_poly, dict):
            return [0.0]

        coeffs = dc_poly.get('coefficients', [[0.0]])

        # Check if coefficients are 2D (list of lists) or 1D (flat list)
        is_2d = isinstance(coeffs, list) and len(coeffs) > 0 and isinstance(coeffs[0], list)

        if not is_2d:
            # Already 1D coefficients
            if isinstance(coeffs, list):
                return coeffs if coeffs else [0.0]
            return [0.0]

        # 2D polynomial: evaluate at mid-azimuth to get 1D in range
        mid_az_pixel = lines / 2.0

        # Evaluate 2D poly at mid_az_pixel to get 1D coefficients in range
        # doppler(rg) = sum_j [sum_i c[i][j] * az^i] * rg^j
        # new_coeff[j] = sum_i c[i][j] * az^i
        try:
            coeffs_2d = np.array(coeffs, dtype=np.float64)
            degree_az = coeffs_2d.shape[0] - 1
            degree_rg = coeffs_2d.shape[1] - 1

            # Compute 1D coefficients by evaluating at mid_az_pixel
            coeffs_1d = []
            for j in range(degree_rg + 1):
                coeff_j = 0.0
                for i in range(degree_az + 1):
                    coeff_j += coeffs_2d[i, j] * (mid_az_pixel ** i)
                coeffs_1d.append(coeff_j)

            # Check if all coefficients are essentially zero
            if all(abs(c) < 1e-15 for c in coeffs_1d):
                return [0.0]

            return coeffs_1d

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse 2D Doppler polynomial: {e}")
            return [0.0]

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

            # Handle both list format [x, y, z] and dict format {x:, y:, z:}
            pos = sv.get('position', [0, 0, 0])
            if isinstance(pos, dict):
                position = [pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)]
            else:
                position = pos

            vel = sv.get('velocity', [0, 0, 0])
            if isinstance(vel, dict):
                velocity = [vel.get('vx', vel.get('x', 0)),
                           vel.get('vy', vel.get('y', 0)),
                           vel.get('vz', vel.get('z', 0))]
            else:
                velocity = vel

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
