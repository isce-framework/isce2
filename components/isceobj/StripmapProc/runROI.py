#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2012 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Author: Brett George
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






import logging
import stdproc
import isceobj
import copy
from mroipac.formimage.FormSLC import FormSLC
import numpy as np 
from isceobj.Location.Peg import Peg
from isceobj.Util.decorators import use_api
import os
import datetime
logger = logging.getLogger('isce.insar.runFormSLC')


@use_api
def focus(frame, outname, amb=0.0):
    from isceobj.Catalog import recordInputsAndOutputs
    from iscesys.ImageUtil.ImageUtil import ImageUtil as IU
    from isceobj.Constants import SPEED_OF_LIGHT

    raw_r0 = frame.startingRange
    raw_dr = frame.getInstrument().getRangePixelSize()
    img = frame.getImage()
    dop = frame._dopplerVsPixel
    #dop = [x/frame.PRF for x in frame._dopplerVsPixel]


    #####Velocity/ acceleration etc
    planet = frame.instrument.platform.planet
    elp =copy.copy( planet.ellipsoid)
    svmid = frame.orbit.interpolateOrbit(frame.sensingMid, method='hermite') 
    xyz = svmid.getPosition()
    vxyz = svmid.getVelocity()
    llh = elp.xyz_to_llh(xyz)

    heading = frame.orbit.getENUHeading(frame.sensingMid)
    print('Heading: ', heading)

    elp.setSCH(llh[0], llh[1], heading)
    sch, schvel = elp.xyzdot_to_schdot(xyz, vxyz)
    vel = np.linalg.norm(schvel)
    hgt = sch[2]
    radius = elp.pegRadCur
  
    ####Computation of acceleration
    dist = np.linalg.norm(xyz)
    r_spinvec = np.array([0., 0., planet.spin])
    r_tempv = np.cross(r_spinvec, xyz)

    inert_acc = np.array([-planet.GM*x/(dist**3) for x in xyz])

    r_tempa = np.cross(r_spinvec, vxyz)
    r_tempvec = np.cross(r_spinvec, r_tempv)

    r_bodyacc = inert_acc - 2 * r_tempa - r_tempvec
    schbasis = elp.schbasis(sch)

    schacc = np.dot(schbasis.xyz_to_sch, r_bodyacc).tolist()[0]


    print('SCH velocity: ', schvel)
    print('SCH acceleration: ', schacc)
    print('Body velocity: ', vel)
    print('Height: ', hgt)
    print('Radius: ', radius)

    #####Setting up formslc
    
    form = FormSLC()
    form.configure()

    ####Width
    form.numberBytesPerLine = img.getWidth()

    ###Includes header
    form.numberGoodBytes = img.getWidth()

    ####First Sample
    form.firstSample = img.getXmin() // 2

    ####Starting range
    form.rangeFirstSample = frame.startingRange

    ####Azimuth looks
    form.numberAzimuthLooks = 1

    ####debug
    form.debugFlag = False

    ####PRF
    form.prf = frame.PRF
    form.sensingStart = frame.sensingStart

    ####Bias
    form.inPhaseValue = frame.getInstrument().inPhaseValue
    form.quadratureValue = frame.getInstrument().quadratureValue

    ####Resolution
    form.antennaLength = frame.instrument.platform.antennaLength
    form.azimuthResolution = 0.6 * form.antennaLength  #85% of max bandwidth

    ####Sampling rate
    form.rangeSamplingRate = frame.getInstrument().rangeSamplingRate

    ####Chirp parameters
    form.chirpSlope =  frame.getInstrument().chirpSlope
    form.rangePulseDuration = frame.getInstrument().pulseLength

    ####Wavelength
    form.radarWavelength = frame.getInstrument().radarWavelength

    ####Secondary range migration
    form.secondaryRangeMigrationFlag = False


    ###pointing direction
    form.pointingDirection = frame.instrument.platform.pointingDirection
    print('Lookside: ', form.pointingDirection)

    ####Doppler centroids
    cfs = [amb, 0., 0., 0.]
    for ii in range(min(len(dop),4)):
        cfs[ii] += dop[ii]/form.prf


    form.dopplerCentroidCoefficients = cfs

    ####Create raw image
    rawimg = isceobj.createRawImage()
    rawimg.load(img.filename + '.xml')
    rawimg.setAccessMode('READ')
    rawimg.createImage()

    form.rawImage = rawimg


    ####All the orbit parameters
    form.antennaSCHVelocity = schvel
    form.antennaSCHAcceleration = schacc
    form.bodyFixedVelocity = vel
    form.spacecraftHeight = hgt
    form.planetLocalRadius = radius



    ###Create SLC image
    slcImg = isceobj.createSlcImage()
    slcImg.setFilename(outname)
    form.slcImage = slcImg

    form.formslc()

    
    ####Populate frame metadata for SLC
    width = form.slcImage.getWidth()
    length = form.slcImage.getLength()
    prf = frame.PRF
    delr = frame.instrument.getRangePixelSize()

    ####Start creating an SLC frame to work with
    slcFrame = copy.deepcopy(frame)

    slcFrame.setStartingRange(form.startingRange)
    slcFrame.setFarRange(form.startingRange + (width-1)*delr)

    tstart = form.slcSensingStart
    tmid = tstart + datetime.timedelta(seconds = 0.5 * length / prf)
    tend = tstart + datetime.timedelta(seconds = (length-1) / prf)

    slcFrame.sensingStart = tstart
    slcFrame.sensingMid = tmid
    slcFrame.sensingStop = tend

    form.slcImage.setAccessMode('READ')
    form.slcImage.setXmin(0)
    form.slcImage.setXmax(width)
    slcFrame.setImage(form.slcImage)

    slcFrame.setNumberOfSamples(width)
    slcFrame.setNumberOfLines(length)

    #####Adjust the doppler polynomial
    dop = frame._dopplerVsPixel[::-1]
    xx = np.linspace(0, (width-1), num=len(dop)+ 1)
    x = (slcFrame.startingRange - frame.startingRange)/delr + xx
    v = np.polyval(dop, x)
    p = np.polyfit(xx, v, len(dop)-1)[::-1]
    slcFrame._dopplerVsPixel = list(p)
    slcFrame._dopplerVsPixel[0] += amb*prf

    return slcFrame



def runFormSLC(self):

    if self._insar.referenceRawProduct is None:
        print('Reference product was unpacked as an SLC. Skipping focusing ....')
        if self._insar.referenceSlcProduct is None:
            raise Exception('However, No reference SLC product found')

    else:
        frame = self._insar.loadProduct(self._insar.referenceRawProduct)
        outdir = os.path.join(self.reference.output + '_slc')
        outname = os.path.join( outdir, os.path.basename(self.reference.output) + '.slc')
        xmlname = outdir + '.xml'
        os.makedirs(outdir, exist_ok=True)

        slcFrame = focus(frame, outname)

        self._insar.referenceGeometrySystem = 'Native Doppler'
        self._insar.saveProduct( slcFrame, xmlname)
        self._insar.referenceSlcProduct = xmlname

        slcFrame = None
        frame = None

    if self._insar.secondaryRawProduct is None:
        print('Secondary product was unpacked as an SLC. Skipping focusing ....')
        if self._insar.secondarySlcProduct is None:
            raise Exception('However, No secondary SLC product found')

    else:
        frame = self._insar.loadProduct(self._insar.secondaryRawProduct)
        outdir = os.path.join(self.secondary.output + '_slc')
        outname = os.path.join( outdir, os.path.basename(self.secondary.output) + '.slc')
        xmlname = outdir + '.xml'
        os.makedirs(outdir, exist_ok=True)

        slcFrame = focus(frame, outname)

        self._insar.secondaryGeometrySystem = 'Native Doppler'
        self._insar.saveProduct( slcFrame, xmlname)
        self._insar.secondarySlcProduct = xmlname

        slcFrame = None
        frame = None

    return None
