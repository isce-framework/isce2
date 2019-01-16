#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Copyright 2013 California Institute of Technology. ALL RIGHTS RESERVED.
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
# Authors: Kosal Khun, Marco Lavalle
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# Comment: Adapted from InsarProc/runFormSLC.py
import logging
import stdproc
import isceobj
from iscesys.ImageUtil.ImageUtil import ImageUtil as IU

logger = logging.getLogger('isce.isceProc.runFormSLC')

def runFormSLC(self):
    """
    Focus raw images.
    """
    infos = {}
    for attribute in ['patchSize', 'numberValidPulses', 'numberPatches', 'numberAzimuthLooks', 'lookSide']:
        infos[attribute] = getattr(self._isce, attribute)
    peg = self._isce.peg
    dopplerCentroid = self._isce.dopplerCentroid
    stdWriter = self._stdWriter
    v, h = self._isce.vh()

    for sceneid in self._isce.selectedScenes:
        self._isce.slcImages[sceneid] = {}
        self._isce.formSLCs[sceneid] = {}
        for pol in self._isce.selectedPols:
            infos['azShiftPixels'] = self._isce.shifts[sceneid][pol]
            orbit = self._isce.orbits[sceneid][pol]
            frame = self._isce.frames[sceneid][pol]
            rawImage = self._isce.iqImages[sceneid][pol]
            catalog = isceobj.Catalog.createCatalog(self._isce.procDoc.name)
            sid = self._isce.formatname(sceneid, pol)
            slcImage, formSlc = run(rawImage, frame, dopplerCentroid, orbit, peg, v, h, infos, stdWriter, catalog=catalog, sceneid=sid)
            ##no need to give outputdir in formslc: extension will be changed from raw to slc
            self._isce.procDoc.addAllFromCatalog(catalog)
            self._isce.formSLCs[sceneid][pol] = formSlc
            self._isce.slcImages[sceneid][pol] = slcImage
    for pol in self._isce.selectedPols:
        formslcs = self._isce.getAllFromPol(pol, self._isce.formSLCs)
        infodict = getinfo(formslcs, sceneid=pol)
        for attribute, value in infodict.items():
            setattr(self._isce, attribute, value)



def run(rawImage, frame, dopplerCentroid, orbit, peg, velocity, height, infos, stdWriter, catalog=None, sceneid='NO_ID'):
    logger.info("Forming SLC: %s" % sceneid)

#    objRaw = rawImage.copy(access_mode='read')
    objRaw = rawImage.clone()
    objRaw.accessMode = 'read'
    objFormSlc = stdproc.createFormSLC()
    objFormSlc.setBodyFixedVelocity(velocity)
    objFormSlc.setSpacecraftHeight(height)
    objFormSlc.setAzimuthPatchSize(infos['patchSize'])
    objFormSlc.setNumberValidPulses(infos['numberValidPulses'])
    objFormSlc.setNumberPatches(infos['numberPatches'])
    objFormSlc.setNumberRangeBin(frame.numberRangeBins)
    objFormSlc.setLookSide(infos['lookSide'])
    objFormSlc.setShift(infos['azShiftPixels'])
    logger.info("Shift in azimuth: %f pixels" % infos['azShiftPixels'])
    objFormSlc.setNumberAzimuthLooks(infos['numberAzimuthLooks'])
    logger.info("Focusing image %s" % sceneid)
    objFormSlc.stdWriter = stdWriter
    objSlc = objFormSlc(rawImage=objRaw,
                        orbit=orbit,
                        frame=frame,
                        planet=frame.instrument.platform.planet,
                        doppler=dopplerCentroid,
                        peg=peg)

    imageSlc = isceobj.createSlcImage()
    IU.copyAttributes(objSlc, imageSlc)
    imageSlc.setAccessMode('read')
    objSlc.finalizeImage()
    objRaw.finalizeImage()
    if catalog is not None:
        isceobj.Catalog.recordInputsAndOutputs(catalog, objFormSlc,
                                               "runFormSLC.%s" % sceneid,
                                               logger,
                                               "runFormSLC.%s" % sceneid)

    logger.info('New Width = %d'%(imageSlc.getWidth()))
    return imageSlc, objFormSlc


def getinfo(formslcs, sceneid='NO_POL'):
    infos = {}
    formSLC1 = formslcs[0]
    numpatches = [ objformslc.numberPatches for objformslc in formslcs ]
    infos['numberPatches'] = min(numpatches)
    infos['is_mocomp'] = int( (formSLC1.azimuthPatchSize - formSLC1.numberValidPulses) / 2 )
    infos['patchSize'] = formSLC1.azimuthPatchSize
    infos['numberValidPulses'] = formSLC1.numberValidPulses
    logger.info('Number of Valid Pulses %s = %d' % (sceneid, formSLC1.numberValidPulses))
    return infos
